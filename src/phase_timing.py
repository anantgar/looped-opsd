"""CUDA-synchronized phase timing."""
import contextlib, time
import torch

class PhaseTimes:
    def __init__(self):self.step={};self.total={}
    def reset_step(self):self.step={}
    @contextlib.contextmanager
    def measure(self,name):
        if torch.cuda.is_available():torch.cuda.synchronize()
        start=time.perf_counter()
        try:yield
        finally:
            if torch.cuda.is_available():torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            self.step[name]=self.step.get(name,0.)+elapsed
            self.total[name]=self.total.get(name,0.)+elapsed
