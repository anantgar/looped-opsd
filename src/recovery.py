"""Atomic, versioned training recovery at optimizer-update boundaries."""
import contextlib
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def fingerprint(config, manifest, metadata):
    sources = [Path(__file__), Path(__file__).with_name('train.py'), Path(__file__).with_name('gsm.py')]
    if config.get('model_family') == 'qwen2-loop':
        sources += [Path(__file__).with_name(name) for name in ['qwen_loop.py', 'qwen_gsm.py', 'qwen_protocol.py']]
        if config.get('protocol') == 'qwen-gsm-last-box-v2':
            sources += [Path(__file__).with_name('answer_extraction.py')]
    if config.get('model_family')=='native-recurrent':
        sources += [Path(__file__).with_name(n) for n in ['native_recurrent.py','native_study.py','qwen_gsm.py','qwen_protocol.py','answer_extraction.py']]
    if config.get('mode')=='adaptive_ce':
        sources += [Path(__file__).with_name('gradient_balance.py')]
    identity = {'config': config, 'data': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                'runtime': {k: metadata[k] for k in ['torch', 'amp_dtype', 'gpu', 'upstream_commit']},
                'source': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class DataCursor:
    def __init__(self, count, batch_size, epochs, seed, state=None):
        self.count, self.batch_size, self.epochs = count, batch_size, epochs
        self.rng = random.Random(seed)
        self.epoch, self.offset, self.order = 0, 0, []
        if state is not None:
            self.epoch, self.offset, self.order = state['epoch'], state['offset'], state['order']
            self.rng.setstate(state['rng'])
            assert sorted(self.order) == list(range(count)) or not self.order
            assert 0 <= self.offset <= count and 0 <= self.epoch <= epochs

    def batch(self):
        if self.order and self.offset == self.count:
            self.epoch += 1
            self.offset, self.order = 0, []
        if self.epoch >= self.epochs:
            return None
        if not self.order:
            self.order = list(range(self.count))
            self.rng.shuffle(self.order)
        return self.order[self.offset:self.offset + self.batch_size]

    def advance(self, size):
        self.offset += size

    def state_dict(self):
        return {'epoch': self.epoch, 'offset': self.offset, 'order': list(self.order),
                'rng': self.rng.getstate()}


def random_state(depth_rng):
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy[0], numpy[1].tolist(),
            numpy[2], numpy[3], numpy[4]], 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'depth': depth_rng.getstate()}


def restore_random(state, depth_rng):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from recovery checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])
    depth_rng.setstate(state['depth'])


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def save(destination, model, optimizer, scaler, cursor, depth_rng, progress, identity):
    atomic_save({'version': 1, 'identity': identity, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'cursor': cursor.state_dict(), 'random': random_state(depth_rng),
        'progress': progress}, destination)


def load(path, model, optimizer, scaler, depth_rng, identity):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != 1 or payload.get('identity') != identity:
        raise ValueError('Recovery checkpoint format/config/data/source/runtime mismatch')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(payload['scaler'])
    restore_random(payload['random'], depth_rng)
    return payload['cursor'], payload['progress']


@contextlib.contextmanager
def exclusive_run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another worker already holds this output directory') from None
        yield
