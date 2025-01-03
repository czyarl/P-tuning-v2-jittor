import math

def _jit_set_profiling_mode(x): pass
def _jit_set_profiling_executor(x): pass
def _jit_override_can_fuse_on_cpu(x): pass
def _jit_override_can_fuse_on_gpu(x): pass

def script(func):
    return func

inf = math.inf
nan = math.nan

## xhb
class NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass