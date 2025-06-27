import torch
import os
import time

def log_gpu_usage(step=None, prefix="", logfile="gpu_memory.log"):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] Step {step if step is not None else '-'} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max: {max_allocated:.2f} MB"
    with open(logfile, "a") as f:
        f.write(f"{prefix}{msg}\n")
    print(f"{prefix}{msg}")