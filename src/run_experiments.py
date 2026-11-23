"""
Main experiment driver.

For every (seed, data fraction, model) configuration it
  1. trains the model from scratch on a stratified subset of CIFAR-10,
  2. measures training time / peak RAM / peak VRAM / GPU energy / CodeCarbon energy,
  3. measures GPU inference on the full 10k test set (batch 1000) and batch-1 latency,
  4. (full data only) measures CPU inference on 2000 test images,
  5. computes Accuracy, macro Precision / Recall / F1 and macro ROC-AUC,
and appends one row to results/raw_runs.csv. The script is resumable: finished
configurations are skipped when it is restarted.

Usage
-----
    python src/run_experiments.py                 # full protocol (5 seeds x 4 fractions x 5 models)
    python src/run_experiments.py --quick         # 1-epoch smoke test
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from complexity import profile_model  # noqa: E402
from data import augment, load_cifar10, normalize, stratified_subset  # noqa: E402
from measure import GPU, Phase, idle_power_baseline, make_tracker  # noqa: E402
from models import MODEL_REGISTRY, build_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
def system_info(gpu: GPU) -> dict:
    import codecarbon
    import sklearn
    import torchvision
    cpu_name = platform.processor()
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
        cpu_name = out.stdout.strip() or cpu_name
    except Exception:
        pass
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu_name,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "codecarbon": codecarbon.__version__,
        "torch_cpu_threads": torch.get_num_threads(),
    }
    info.update(gpu.info())
    return info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    prob = torch.softmax(logits.float(), 1).cpu().numpy()
    pred = prob.argmax(1)
    yt = y.cpu().numpy()
    return {
        "accuracy": accuracy_score(yt, pred),
        "precision_macro": precision_score(yt, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(yt, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc_score(yt, prob, multi_class="ovr", average="macro"),
    }


@torch.inference_mode()
def predict(model, x_uint8, batch, device="cuda", amp=True):
    outs = []
    for i in range(0, x_uint8.shape[0], batch):
        xb = normalize(x_uint8[i:i + batch])
        if device == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            outs.append(model(xb))
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
def run_one(cfg, data, gpu, tracker, model_name, fraction, seed):
    dev = "cuda"
    set_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    idx = stratified_subset(data["y_train"], fraction, seed)
    n = len(idx)
    x_tr, y_tr = data["x_train"], data["y_train"]

    model = build_model(model_name).to(dev, memory_format=torch.channels_last)
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, epochs=cfg.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.25)
    loss_fn = nn.CrossEntropyLoss()
    temp_start = gpu.temperature()

    # ------------------------------ TRAINING ------------------------------ #
    epoch_losses = []
    model.train()
    with Phase(gpu, tracker, "train") as ph_train:
        for _ in range(cfg.epochs):
            perm = idx[torch.randperm(n, device=dev, generator=gen)]
            loss_sum = torch.zeros((), device=dev)
            for b in range(steps_per_epoch):
                bi = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                xb = normalize(augment(x_tr[bi], gen)).contiguous(memory_format=torch.channels_last)
                yb = y_tr[bi]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.detach() * bi.numel()
            epoch_losses.append(loss_sum)
    epoch_losses = [round(float(l) / n, 4) for l in epoch_losses]
    tr = ph_train.result

    # --------------------------- GPU INFERENCE ---------------------------- #
    model.eval()
    x_te, y_te = data["x_test"], data["y_test"]
    m_test = x_te.shape[0]
    predict(model, x_te[:2000], cfg.infer_batch)  # warm-up (cuDNN autotune)
    passes = 0
    with Phase(gpu, tracker, "infer_gpu") as ph_inf:
        t_start = time.perf_counter()
        while passes < 2 or time.perf_counter() - t_start < cfg.min_infer_seconds:
            logits = predict(model, x_te, cfg.infer_batch)
            torch.cuda.synchronize()
            passes += 1
    inf = ph_inf.result
    metrics = classification_metrics(logits, y_te)

    # batch-1 latency (online / real-time setting)
    lat_n = cfg.latency_samples
    with torch.inference_mode():
        for i in range(10):
            predict(model, x_te[i:i + 1], 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(lat_n):
            predict(model, x_te[i:i + 1], 1)
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / lat_n * 1000

    row = {
        "model": model_name, "fraction": fraction, "seed": seed, "n_train": n,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "gpu_temp_start_c": temp_start,
        **{f"train_{k}": v for k, v in tr.items()},
        "infer_passes": passes,
        "infer_time_s": inf["time_s"] / passes,
        "infer_gpu_energy_j": inf["gpu_energy_j"] / passes,
        "infer_gpu_power_avg_w": inf["gpu_power_avg_w"],
        "infer_gpu_util_avg": inf["gpu_util_avg"],
        "infer_peak_ram_mb": inf["peak_ram_mb"],
        "infer_peak_vram_torch_mb": inf["peak_vram_torch_mb"],
        "infer_peak_vram_nvml_mb": inf["peak_vram_nvml_mb"],
        "infer_throughput_img_s": m_test * passes / inf["time_s"],
        "latency_b1_ms": latency_ms,
        **metrics,
        "loss_curve": json.dumps(epoch_losses),
    }
    for k in ("cc_cpu_energy_j", "cc_gpu_energy_j", "cc_ram_energy_j", "cc_total_energy_j", "cc_emissions_g"):
        if k in inf:
            row[f"infer_{k}"] = inf[k] / passes

    # --------------------------- CPU INFERENCE ---------------------------- #
    if cfg.cpu_infer and fraction >= 1.0:
        model_cpu = model.float().to("cpu", memory_format=torch.contiguous_format)
        x_cpu = data["x_test"][:cfg.cpu_infer_samples].cpu()
        predict(model_cpu, x_cpu[:cfg.cpu_infer_batch], cfg.cpu_infer_batch, device="cpu")  # warm-up
        with Phase(gpu, tracker, "infer_cpu", device="cpu") as ph_cpu:
            predict(model_cpu, x_cpu, cfg.cpu_infer_batch, device="cpu")
        c = ph_cpu.result
        row.update({
            "cpu_infer_time_s": c["time_s"],
            "cpu_infer_ms_per_img": c["time_s"] / cfg.cpu_infer_samples * 1000,
            "cpu_infer_peak_ram_mb": c["peak_ram_mb"],
            "cpu_infer_ram_delta_mb": c["ram_delta_mb"],
            "cpu_infer_util_avg": c["cpu_util_avg"],
            "cpu_infer_cc_cpu_energy_j": c.get("cc_cpu_energy_j", float("nan")),
            "cpu_infer_cc_total_energy_j": c.get("cc_total_energy_j", float("nan")),
        })
        del model_cpu
    else:  # keep a fixed CSV schema
        for k in ("cpu_infer_time_s", "cpu_infer_ms_per_img", "cpu_infer_peak_ram_mb", "cpu_infer_ram_delta_mb", "cpu_infer_util_avg",
                  "cpu_infer_cc_cpu_energy_j", "cpu_infer_cc_total_energy_j"):
            row[k] = float("nan")
    del model, opt
    torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--infer-batch", type=int, default=1000)
    ap.add_argument("--min-infer-seconds", type=float, default=3.0)
    ap.add_argument("--latency-samples", type=int, default=300)
    ap.add_argument("--cpu-infer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu-infer-samples", type=int, default=2000)
    ap.add_argument("--cpu-infer-batch", type=int, default=250)
    ap.add_argument("--pause", type=float, default=5.0, help="cool-down seconds between runs")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, 1 seed, 2 fractions")
    cfg = ap.parse_args()
    if cfg.quick:
        cfg.epochs, cfg.seeds, cfg.fractions = 1, [0], [0.1, 1.0]
        cfg.out = os.path.join(ROOT, "results_quick")
        cfg.min_infer_seconds, cfg.latency_samples, cfg.pause = 1.0, 50, 0

    assert torch.cuda.is_available(), "CUDA GPU is required for this protocol"
    os.makedirs(cfg.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    gpu = GPU(0)

    info = system_info(gpu)
    info["protocol"] = {k: v for k, v in vars(cfg).items()}
    print(json.dumps(info, indent=2))

    # ---- theoretical complexity of each architecture ----
    comp_rows = []
    for name in MODEL_REGISTRY:
        p = profile_model(build_model(name))
        p["model"] = name
        comp_rows.append(p)
    with open(os.path.join(cfg.out, "model_complexity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + [k for k in comp_rows[0] if k != "model"])
        w.writeheader()
        w.writerows(comp_rows)

    data = load_cifar10(os.path.join(ROOT, "data"))
    print("Measuring idle GPU power for 10 s ...")
    info.update(idle_power_baseline(gpu, 10.0))
    with open(os.path.join(cfg.out, "system_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    tracker = make_tracker(cfg.out)

    # warm-up run (CUDA context, cuDNN autotuning) - not recorded
    print("Warm-up run (not recorded) ...")
    wcfg = argparse.Namespace(**{**vars(cfg), "epochs": 1, "cpu_infer": False, "min_infer_seconds": 0.5,
                                 "latency_samples": 10})
    for name in cfg.models:
        run_one(wcfg, data, gpu, tracker, name, 0.1, 999)

    raw_path = os.path.join(cfg.out, "raw_runs.csv")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for r in csv.DictReader(f):
                done.add((r["model"], float(r["fraction"]), int(r["seed"])))

    jobs = []
    for seed in cfg.seeds:
        for frac in cfg.fractions:
            order = list(cfg.models)
            random.Random(seed * 100 + int(frac * 100)).shuffle(order)  # avoid systematic thermal bias
            jobs += [(m, frac, seed) for m in order]
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} configurations, {len(todo)} remaining")

    fieldnames = None
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            fieldnames = next(csv.reader(f), None)
    t_begin = time.time()
    for k, (m, frac, seed) in enumerate(todo, 1):
        row = run_one(cfg, data, gpu, tracker, m, frac, seed)
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        if fieldnames is None:
            fieldnames = list(row.keys())
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
        new_file = not os.path.exists(raw_path)
        with open(raw_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)
        el = time.time() - t_begin
        eta = el / k * (len(todo) - k)
        print(f"[{k:3d}/{len(todo)}] {m:12s} frac={frac:<5} seed={seed} n={row['n_train']:5d} "
              f"train={row['train_time_s']:7.1f}s E_gpu={row['train_gpu_energy_j']:8.0f}J "
              f"acc={row['accuracy']:.4f} VRAM={row['train_peak_vram_torch_mb']:6.0f}MB "
              f"| ETA {eta/60:5.1f} min", flush=True)
        time.sleep(cfg.pause)

    if tracker is not None:
        try:
            tracker.stop()
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()
