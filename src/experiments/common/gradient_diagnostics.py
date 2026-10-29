"""Exact component cosine with one disk snapshot, bounded RAM/GPU temporaries.

The caller accumulates each full-effective-batch component at unchanged weights.
No autograd graphs are retained, no extra optimizer step is performed, and the
original AMP scale is preserved until the combined gradient is unscaled once.
"""
import math
import shutil
from pathlib import Path
import torch


class ComponentSnapshot:
    def __init__(self, parameters, folder):
        self.parameters = list(parameters)
        self.folder = Path(folder)
        self.saved = {}
        if self.folder.exists():
            raise RuntimeError('Diagnostic folder already exists; never overwrite it')
        self.folder.mkdir(parents=True)

    def capture(self):
        """Store the shallow gradient, then free parameter.grad for the anchor."""
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is not None:
                gradient = parameter.grad.detach().cpu()
                torch.save(gradient, self.folder / f'{i}.pt')
                self.saved[i] = True
                del gradient
            parameter.grad = None

    def combine(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid AMP scale')
        a2 = b2 = dot = 0.0
        for i, parameter in enumerate(self.parameters):
            shallow = torch.load(self.folder / f'{i}.pt', map_location='cpu',
                                 weights_only=True) if i in self.saved else None
            anchor = parameter.grad
            if shallow is None and anchor is None:
                continue
            # FP64 reduction on small chunks only. Preserve FP32 gradient dtype.
            for start in range(0, parameter.numel(), 1 << 18):
                a = shallow.reshape(-1)[start:start+(1 << 18)].double() if shallow is not None else None
                b = anchor.detach().reshape(-1)[start:start+(1 << 18)].cpu().double() if anchor is not None else None
                if a is not None: a2 += float(torch.dot(a, a))
                if b is not None: b2 += float(torch.dot(b, b))
                if a is not None and b is not None: dot += float(torch.dot(a, b))
            if shallow is not None:
                if anchor is None:
                    parameter.grad = shallow.to(parameter.device)
                else:
                    # Transfer one chunk at a time, avoiding a second GPU vector.
                    with torch.no_grad():
                        flat = anchor.view(-1)
                        for start in range(0, parameter.numel(), 1 << 18):
                            flat[start:start+(1 << 18)].add_(shallow.reshape(-1)[start:start+(1 << 18)].to(parameter.device))
            del shallow
        if not all(math.isfinite(v) for v in [a2, b2, dot]):
            raise FloatingPointError('Nonfinite component gradient diagnostic')
        defined = a2 > 0 and b2 > 0
        cosine = max(-1., min(1., dot / math.sqrt(a2*b2))) if defined else None
        return {'gradient_shallow_norm': math.sqrt(a2)/scale,
                'gradient_anchor_norm': math.sqrt(b2)/scale,
                'gradient_cosine': cosine,
                'gradient_cosine_defined': defined,
                'gradient_combined_norm_estimate': math.sqrt(max(0., a2+b2+2*dot))/scale,
                'gradient_probe_scope': 'full effective batch; trainable parameters; before clipping',
                'gradient_probe_method': 'exact streamed FP64 dot/norm; disk FP32 snapshot'}

    def close(self):
        shutil.rmtree(self.folder)
