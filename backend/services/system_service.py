from __future__ import annotations

import platform
import subprocess
import psutil


class SystemService:
    """Provides system usage metrics. Single Responsibility: metrics only."""

    @staticmethod
    def cpu_percent() -> float:
        return float(psutil.cpu_percent())

    @staticmethod
    def memory_stats() -> dict:
        mem = psutil.virtual_memory()
        return {
            "memory_percent": float(mem.percent),
            "memory_available_gb": round(mem.available / (1024**3), 2),
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
        }

    @staticmethod
    def gpu_percent() -> float:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
            pynvml.nvmlShutdown()
            return gpu_util
        except Exception:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
                return float(out.strip().splitlines()[0])
            except Exception:
                return 0.0

    @staticmethod
    def npu_percent() -> float:
        if platform.system() == "Windows":
            try:
                cmd = (
                    "Get-Counter '\\AI Accelerator(*)\\Usage Percentage' "
                    "| Select -First 1 -ExpandProperty CounterSamples "
                    "| Select -ExpandProperty CookedValue"
                )
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
                return float(out.strip())
            except Exception:
                return 0.0
        return 0.0


