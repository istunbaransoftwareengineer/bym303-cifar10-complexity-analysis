"""
Measurement utilities: wall-clock time, peak RAM, peak VRAM, GPU energy (NVML),
and whole-system energy / carbon estimates (CodeCarbon).

How each quantity is obtained
-----------------------------
Time   : time.perf_counter() around the phase, with torch.cuda.synchronize()
         before reading the clock (GPU kernels are asynchronous).
RAM    : a background thread samples the process RSS (psutil) every 50 ms and
         keeps the maximum -> peak RAM of the Python process.
VRAM   : torch.cuda.max_memory_allocated() (tensor memory requested by PyTorch,
         net of the dataset tensors that are already resident) and the NVML
         device memory in use (includes CUDA context + caching allocator).
Energy : (1) GPU  - NVML hardware energy counter
                    nvmlDeviceGetTotalEnergyConsumption() [mJ]; if the counter
                    is not supported the sampled power is integrated (trapezoid).
         (2) CPU/RAM/GPU + CO2 - CodeCarbon OfflineEmissionsTracker using the
                    Turkish grid carbon intensity (country_iso_code='TUR').
         E = P_avg * T is therefore *measured*, not assumed: P_avg = E / T is
         reported separately so that the energy is not reduced to time only.
"""
import os
import threading
import time
import warnings

import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False


class GPU:
    """Thin NVML wrapper (device 0)."""

    def __init__(self, index: int = 0):
        self.ok = _NVML and torch.cuda.is_available()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index) if self.ok else None
        self.has_energy_counter = False
        if self.ok:
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.has_energy_counter = True
            except Exception:
                self.has_energy_counter = False

    def power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 if self.ok else 0.0

    def energy_mj(self) -> float:
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)) if self.has_energy_counter else 0.0

    def util(self) -> float:
        return float(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu) if self.ok else 0.0

    def temperature(self) -> float:
        return float(pynvml.nvmlDeviceGetTemperature(self.handle, 0)) if self.ok else 0.0

    def mem_used_mb(self) -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20 if self.ok else 0.0

    def info(self) -> dict:
        if not self.ok:
            return {}
        return {
            "gpu_name": pynvml.nvmlDeviceGetName(self.handle),
            "gpu_driver": pynvml.nvmlSystemGetDriverVersion(),
            "gpu_mem_total_mb": pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / 2**20,
            "gpu_power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0,
            "gpu_energy_counter": self.has_energy_counter,
        }


class Sampler(threading.Thread):
    """Samples GPU power/util, process RSS and CPU load every `interval` seconds."""

    def __init__(self, gpu: GPU, interval: float = 0.05):
        super().__init__(daemon=True)
        self.gpu, self.interval = gpu, interval
        self.proc = psutil.Process(os.getpid())
        self._halt = threading.Event()
        self.t, self.p, self.u, self.rss, self.cpu, self.vram = [], [], [], [], [], []
        psutil.cpu_percent(None)

    def run(self):
        while not self._halt.is_set():
            self.t.append(time.perf_counter())
            self.p.append(self.gpu.power_w())
            self.u.append(self.gpu.util())
            self.vram.append(self.gpu.mem_used_mb())
            self.rss.append(self.proc.memory_info().rss / 2**20)
            self.cpu.append(psutil.cpu_percent(None))
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join()

    def integrated_energy_j(self) -> float:
        e = 0.0
        for i in range(1, len(self.t)):
            e += 0.5 * (self.p[i] + self.p[i - 1]) * (self.t[i] - self.t[i - 1])
        return e


def _mean(x):
    return sum(x) / len(x) if x else 0.0


class Phase:
    """
    Context manager that measures one phase (training or inference).

        with Phase(gpu, tracker, "train") as ph:
            ...work...
        ph.result -> dict
    """

    def __init__(self, gpu: GPU, tracker=None, name: str = "phase", device: str = "cuda"):
        self.gpu, self.tracker, self.name, self.device = gpu, tracker, name, device
        self.result = {}

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.alloc0 = torch.cuda.memory_allocated()
        self.rss0 = psutil.Process().memory_info().rss / 2**20
        self.sampler = Sampler(self.gpu)
        self.sampler.start()
        if self.tracker is not None:
            self.tracker.start_task(self.name)
        self.e0 = self.gpu.energy_mj()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e1 = self.gpu.energy_mj()
        cc = None
        if self.tracker is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cc = self.tracker.stop_task()
        self.sampler.stop()
        s = self.sampler
        elapsed = t1 - self.t0
        gpu_j = (e1 - self.e0) / 1000.0 if self.gpu.has_energy_counter else s.integrated_energy_j()
        r = {
            "time_s": elapsed,
            "gpu_energy_j": gpu_j,
            "gpu_power_avg_w": gpu_j / elapsed if elapsed > 0 else 0.0,
            "gpu_power_max_w": max(s.p) if s.p else 0.0,
            "gpu_util_avg": _mean(s.u),
            "cpu_util_avg": _mean(s.cpu),
            "peak_ram_mb": max(s.rss) if s.rss else psutil.Process().memory_info().rss / 2**20,
            "peak_vram_nvml_mb": max(s.vram) if s.vram else 0.0,
        }
        r["ram_delta_mb"] = r["peak_ram_mb"] - self.rss0
        if self.device == "cuda":
            r["peak_vram_torch_mb"] = (torch.cuda.max_memory_allocated() - self.alloc0) / 2**20
            r["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        if cc is not None:
            r.update({
                "cc_cpu_energy_j": cc.cpu_energy * 3.6e6,
                "cc_gpu_energy_j": cc.gpu_energy * 3.6e6,
                "cc_ram_energy_j": cc.ram_energy * 3.6e6,
                "cc_total_energy_j": cc.energy_consumed * 3.6e6,
                "cc_emissions_g": cc.emissions * 1000.0,
            })
        self.result = r
        return False


def make_tracker(output_dir: str):
    """CodeCarbon tracker in offline mode with Turkey's grid carbon intensity."""
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            country_iso_code="TUR",
            measure_power_secs=1,
            log_level="error",
            save_to_file=False,
            output_dir=output_dir,
            tracking_mode="process",
        )
        tracker.start()
        return tracker
    except Exception as e:  # pragma: no cover
        print(f"[warn] CodeCarbon unavailable: {e}")
        return None


def idle_power_baseline(gpu: GPU, seconds: float = 10.0) -> dict:
    """Average GPU power while idle - used to separate dynamic from static energy."""
    torch.cuda.synchronize()
    s = Sampler(gpu, interval=0.1)
    s.start()
    time.sleep(seconds)
    s.stop()
    return {"gpu_idle_power_w": _mean(s.p), "cpu_idle_util": _mean(s.cpu)}
