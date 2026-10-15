import importlib
import importlib.metadata
import platform
import subprocess
import sys
import psutil


def doctor():
    info=dict(os=platform.platform(),python=sys.version,executable=sys.executable,
              cpu_model=platform.processor(),physical_cores=psutil.cpu_count(logical=False),
              logical_threads=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,
              wsl="microsoft" in platform.release().lower(),packages={})
    try:
        if platform.system()=="Windows":
            info["cpu_model"]=subprocess.check_output(["powershell","-NoProfile","-Command","(Get-CimInstance Win32_Processor).Name"],text=True,timeout=10).strip()
        elif platform.system()=="Linux":
            with open("/proc/cpuinfo") as f:
                info["cpu_model"]=next(line.split(":",1)[1].strip() for line in f if line.startswith("model name"))
    except Exception: pass
    for name in ("numpy","scipy","numba","torch","triton","statesegmentation"):
        try: info["packages"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: info["packages"][name]=None
    try:
        info["nvidia_smi"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,compute_cap","--format=csv"],text=True,timeout=10).strip()
    except Exception as e: info["nvidia_smi"]=str(e)
    if platform.system()=="Windows":
        try:
            raw=subprocess.check_output(["wsl","--list","--verbose"],timeout=10).decode("utf-16le",errors="replace")
            info["wsl_distributions_state_version"]=raw.splitlines()[1:]
        except Exception as e: info["wsl_status"]=str(e)
    try:
        import torch
        info["cuda_available"]=torch.cuda.is_available(); info["cuda_runtime"]=torch.version.cuda
        if info["cuda_available"]:
            p=torch.cuda.get_device_properties(0)
            info["gpu"]=dict(name=p.name,compute_capability=list(torch.cuda.get_device_capability()),total_memory=p.total_memory,
                             fp64_supported=True,fp32_supported=True,tf32_supported=p.major>=8,
                             fp64_throughput="Not inferred from dtype support; measure with benchmarks/precision_bench.py")
    except Exception as e: info["cuda_available"]=False; info["torch_error"]=str(e)
    return info
