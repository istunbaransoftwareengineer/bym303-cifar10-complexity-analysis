"""
Statistical analysis, figures and LaTeX tables from results/raw_runs.csv.

    python src/analyze.py            # -> results/summary_*.csv, figures/*.pdf|png, paper/tables/*.tex

Main analyses
-------------
* mean +- std over the 5 seeds for every (model, data fraction)
* empirical scaling exponent b in  T(n) = c * n^b  (log-log least squares, 95% CI)
  -> compared with the theoretical O(n) training cost
* theoretical FLOPs (3 * 2 * MACs * n * E) vs measured time -> achieved TFLOP/s
* average power P_avg = E / T  -> shows that energy is NOT just time
* measured peak VRAM vs analytical memory model
* accuracy / energy trade-off and Pareto front
"""
import json
import os
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from complexity import theoretical_train_memory_mb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper")
TAB = os.path.join(PAPER, "tables")
for d in (FIG, TAB, os.path.join(PAPER, "figures")):
    os.makedirs(d, exist_ok=True)

ORDER = ["MLP", "SmallCNN", "MediumCNN", "MobileNetV2", "ResNet18"]
# validated categorical palette (fixed order, never cycled) + secondary encoding by marker
COLORS = {"MLP": "#2a78d6", "SmallCNN": "#eb6834", "MediumCNN": "#1baf7a",
          "MobileNetV2": "#eda100", "ResNet18": "#e87ba4"}
MARKERS = {"MLP": "o", "SmallCNN": "s", "MediumCNN": "^", "MobileNetV2": "D", "ResNet18": "v"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TICK = {"MLP": "MLP", "SmallCNN": "Small\nCNN", "MediumCNN": "Medium\nCNN", "MobileNetV2": "Mobile\nNetV2", "ResNet18": "ResNet\n18"}
ERR = dict(ecolor="#52514e", elinewidth=0.8, capsize=2)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
COL_W = 3.45  # IEEE single column width in inches


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "pdf":
            shutil.copy(p, os.path.join(PAPER, "figures", f"{name}.pdf"))
    plt.close(fig)


def end_labels(ax, items, logy=True, min_gap=0.075):
    """Direct labels at line ends, pushed apart vertically so they never collide.
    items: list of (x, y, text). min_gap is a fraction of the axis height."""
    if not items:
        return
    ax.figure.canvas.draw()
    lo, hi = ax.get_ylim()
    f = (lambda v: np.log10(v)) if logy else (lambda v: v)
    finv = (lambda v: 10 ** v) if logy else (lambda v: v)
    span = f(hi) - f(lo)
    its = sorted(items, key=lambda t: t[1])
    ys = [f(t[1]) for t in its]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap * span)
    for (x, y, txt), yy in zip(its, ys):
        ax.annotate(txt, xy=(x, y), xytext=(x * 1.08, finv(yy)), textcoords="data",
                    va="center", fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if abs(yy - f(y)) > 0.01 * span else None)


def pm(m, s, fmt="{:.2f}"):
    return f"{fmt.format(m)} $\\pm$ {fmt.format(s)}"


# --------------------------------------------------------------------------- #
def load():
    raw = pd.read_csv(os.path.join(RES, "raw_runs.csv"))
    comp = pd.read_csv(os.path.join(RES, "model_complexity.csv")).set_index("model")
    with open(os.path.join(RES, "system_info.json")) as f:
        info = json.load(f)
    raw = raw[raw["model"].isin(ORDER)].copy()
    raw["macs"] = raw["model"].map(comp["macs_per_sample"])
    raw["params"] = raw["model"].map(comp["params"])
    # theoretical training work: forward+backward ~ 3x forward, 2 FLOPs per MAC
    raw["train_tflop"] = 3 * 2 * raw["macs"] * raw["n_train"] * raw["epochs"] / 1e12
    raw["achieved_tflops"] = raw["train_tflop"] / raw["train_time_s"]
    raw["train_gpu_energy_kj"] = raw["train_gpu_energy_j"] / 1000
    if "train_cc_total_energy_j" in raw:
        raw["train_total_energy_kj"] = raw["train_cc_total_energy_j"] / 1000
    raw["infer_energy_mj_per_img"] = raw["infer_gpu_energy_j"] / 10000 * 1000
    raw["time_per_sample_epoch_us"] = raw["train_time_s"] / (raw["n_train"] * raw["epochs"]) * 1e6
    raw["energy_per_sample_epoch_mj"] = raw["train_gpu_energy_j"] / (raw["n_train"] * raw["epochs"]) * 1e3
    raw["dyn_energy_kj"] = (raw["train_gpu_energy_j"] - info.get("gpu_idle_power_w", 0) * raw["train_time_s"]) / 1000
    return raw, comp, info


def summarize(raw):
    num = raw.select_dtypes(include=[np.number]).columns.drop(["seed", "fraction"], errors="ignore")
    g = raw.groupby(["model", "fraction"])[list(num)]
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).reset_index()
    out["n_runs"] = raw.groupby(["model", "fraction"]).size().values
    out["model"] = pd.Categorical(out["model"], ORDER, ordered=True)
    return out.sort_values(["model", "fraction"])


def scaling_fit(raw, col):
    rows = []
    for m in ORDER:
        d = raw[raw.model == m]
        if d["fraction"].nunique() < 2:
            continue
        r = stats.linregress(np.log(d["n_train"]), np.log(d[col]))
        tcrit = stats.t.ppf(0.975, len(d) - 2)
        rows.append({"model": m, "metric": col, "exponent_b": r.slope,
                     "ci95_low": r.slope - tcrit * r.stderr, "ci95_high": r.slope + tcrit * r.stderr,
                     "r2": r.rvalue ** 2, "coef_c": np.exp(r.intercept), "n_points": len(d)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_scaling(raw, summ, fits, col, ylabel, name, title):
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    labels = []
    for m in ORDER:
        s = summ[summ.model == m]
        if s.empty:
            continue
        x, y, e = s["n_train_mean"], s[f"{col}_mean"], s[f"{col}_std"]
        ax.errorbar(x, y, yerr=e, color=COLORS[m], marker=MARKERS[m], ms=4.5, capsize=2,
                    lw=1.4, mec="white", mew=0.6, label=None)
        f = fits[fits.model == m]
        if not f.empty:
            labels.append((x.iloc[-1], y.iloc[-1], f"{m}  b={f.iloc[0]['exponent_b']:.2f}"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    end_labels(ax, labels, logy=True)
    ax.set_xlabel("Training-set size n (images, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xticks([5000, 12500, 25000, 50000])
    ax.set_xticklabels(["5k", "12.5k", "25k", "50k"])
    ax.set_xlim(4000, 160000)
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=4.5, mec="white", lw=1.4, label=m)
               for m in ORDER]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=1)
    save(fig, name)


def fig_accuracy(summ):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    labels = []
    for m in ORDER:
        s = summ[summ.model == m]
        ax.errorbar(s["n_train_mean"], s["accuracy_mean"] * 100, yerr=s["accuracy_std"] * 100,
                    color=COLORS[m], marker=MARKERS[m], ms=4.5, capsize=2, mec="white", mew=0.6, label=m)
        labels.append((s["n_train_mean"].iloc[-1], s["accuracy_mean"].iloc[-1] * 100,
                       f"{m} {s['accuracy_mean'].iloc[-1]*100:.1f}%"))
    ax.set_xscale("log")
    ax.set_xlim(4000, 110000)
    ax.set_ylim(30, 104)
    end_labels(ax, labels, logy=False, min_gap=0.075)
    ax.set_xticks([5000, 12500, 25000, 50000])
    ax.set_xticklabels(["5k", "12.5k", "25k", "50k"])
    ax.set_xlim(4000, 110000)
    ax.set_xlabel("Training-set size n (images, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Learning curves (mean $\\pm$ std, 5 seeds)", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6)
    save(fig, "fig3_accuracy_vs_n")


def pareto(points):
    """points: list of (cost, acc, name) -> names on the Pareto front (min cost, max acc)."""
    front, best = [], -1
    for c, a, n in sorted(points):
        if a > best:
            front.append((c, a, n))
            best = a
    return front


def fig_tradeoff(summ, comp):
    full = summ[summ.fraction == 1.0]
    col = "train_total_energy_kj" if "train_total_energy_kj_mean" in full else "train_gpu_energy_kj"
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    pts = []
    for m in ORDER:
        s = full[full.model == m]
        if s.empty:
            continue
        x, y = s[f"{col}_mean"].iloc[0], s["accuracy_mean"].iloc[0] * 100
        xe, ye = s[f"{col}_std"].iloc[0], s["accuracy_std"].iloc[0] * 100
        size = 20 + 160 * np.sqrt(comp.loc[m, "params"] / comp["params"].max())
        ax.scatter(x, y, s=size, color=COLORS[m], marker=MARKERS[m], edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(x, y, xerr=xe, yerr=ye, color=COLORS[m], lw=0.8, capsize=1.5, zorder=2)
        off, ha, va = {"ResNet18": ((-8, 8), "right", "bottom"),
                       "MobileNetV2": ((6, -6), "left", "top")}.get(m, ((6, -2), "left", "top"))
        ax.annotate(f"{m}\n{comp.loc[m, 'params']/1e6:.2f}M params", (x, y), xytext=off,
                    textcoords="offset points", fontsize=6.2, color=INK2, va=va, ha=ha)
        pts.append((x, y, m))
    fr = pareto(pts)
    ax.plot([p[0] for p in fr], [p[1] for p in fr], ls="--", color=MUTED, lw=0.9, zorder=1,
            label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Training energy (kJ, log scale; CodeCarbon CPU+GPU+RAM)" if col.startswith("train_total")
                  else "Training GPU energy (kJ, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. energy at n = 50k (marker area ~ parameters)", loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.margins(x=0.25, y=0.12)
    ax.set_ylim(top=99)
    save(fig, "fig4_tradeoff_accuracy_energy")
    return [p[2] for p in fr]


def fig_memory(summ, comp, batch_size):
    full = summ[summ.fraction == 1.0].set_index("model")
    ms = [m for m in ORDER if m in full.index]
    meas = [full.loc[m, "train_peak_vram_torch_mb_mean"] for m in ms]
    err = [full.loc[m, "train_peak_vram_torch_mb_std"] for m in ms]
    theo = [theoretical_train_memory_mb(comp.loc[m, "params"], comp.loc[m, "activations_per_sample"],
                                        batch_size) for m in ms]
    inf = [full.loc[m, "infer_peak_vram_torch_mb_mean"] for m in ms]
    x = np.arange(len(ms))
    w = 0.27
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(x - w, theo, w - 0.02, color="#86b6ef", label="Analytical estimate (training)")
    ax.bar(x, meas, w - 0.02, yerr=err, color="#2a78d6", label="Measured peak (training)", error_kw=ERR)
    ax.bar(x + w, inf, w - 0.02, color="#184f95", label="Measured peak (inference, B=1000)")
    for xi, v in zip(x, meas):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[m] for m in ms], fontsize=6.5)
    ax.set_ylim(0, max(meas + inf + theo) * 1.45)
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title(f"Peak VRAM attributable to the model (B={batch_size})", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_memory_vram")
    return dict(zip(ms, theo))


def fig_efficiency(summ, info):
    full = summ[summ.fraction == 1.0].set_index("model")
    ms = [m for m in ORDER if m in full.index]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.05, 2.3))
    ax = axes[0]
    v = [full.loc[m, "achieved_tflops_mean"] for m in ms]
    ax.bar([TICK[m] for m in ms], v, color=[COLORS[m] for m in ms], width=0.6)
    for i, val in enumerate(v):
        ax.annotate(f"{val:.2f}", (i, val), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=6.5, color=INK2)
    ax.set_ylabel("Achieved TFLOP/s (theoretical FLOPs / time)")
    ax.set_title("(a) Hardware utilisation during training", loc="left")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=6.5)
    ax = axes[1]
    p = [full.loc[m, "train_gpu_power_avg_w_mean"] for m in ms]
    pe = [full.loc[m, "train_gpu_power_avg_w_std"] for m in ms]
    ax.bar([TICK[m] for m in ms], p, yerr=pe, color=[COLORS[m] for m in ms], width=0.6, error_kw=ERR)
    idle = info.get("gpu_idle_power_w")
    if idle:
        ax.axhline(idle, color=MUTED, ls="--", lw=0.9)
        ax.annotate(f"idle {idle:.1f} W", (len(ms) - 0.5, idle), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK2)
    for i, (val, er) in enumerate(zip(p, pe)):
        ax.annotate(f"{val:.1f}", (i, val + er), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=6.5, color=INK2)
    ax.set_ylabel("Average GPU power $P_{avg}=E/T$ (W)")
    ax.set_title("(b) Average GPU power during training", loc="left")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout()
    save(fig, "fig6_efficiency_power")


def fig_inference(summ):
    full = summ[summ.fraction == 1.0].set_index("model")
    ms = [m for m in ORDER if m in full.index]
    gpu_b = [full.loc[m, "infer_time_s_mean"] / 10000 * 1000 for m in ms]           # ms / image, B=1000
    gpu_1 = [full.loc[m, "latency_b1_ms_mean"] for m in ms]                         # ms / image, B=1
    cpu_b = [full.loc[m, "cpu_infer_ms_per_img_mean"] if "cpu_infer_ms_per_img_mean" in full else np.nan
             for m in ms]
    x = np.arange(len(ms))
    w = 0.27
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(x - w, gpu_b, w - 0.02, color="#2a78d6", label="GPU, batch 1000 (throughput)")
    ax.bar(x, cpu_b, w - 0.02, color="#eb6834", label="CPU, batch 250")
    ax.bar(x + w, gpu_1, w - 0.02, color="#1baf7a", label="GPU, batch 1 (latency)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[m] for m in ms], fontsize=6.5)
    ax.set_ylabel("Time per image (ms, log scale)")
    ax.set_title("Inference cost: GPU vs CPU, batched vs online", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(top=max(np.nanmax(cpu_b), max(gpu_1)) * 30)
    save(fig, "fig7_inference")


def fig_time_vs_flops(summ):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for m in ORDER:
        s = summ[summ.model == m]
        ax.errorbar(s["train_tflop_mean"], s["train_time_s_mean"], yerr=s["train_time_s_std"],
                    color=COLORS[m], marker=MARKERS[m], ms=4.5, mec="white", mew=0.6, capsize=2, label=m)
    xs = np.logspace(np.log10(summ["train_tflop_mean"].min()), np.log10(summ["train_tflop_mean"].max()), 10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Theoretical training work 6$\\cdot$F$\\cdot$n$\\cdot$E (TFLOP, log)")
    ax.set_ylabel("Measured training time (s, log)")
    ax.set_title("Theory vs. measurement: time against FLOPs", loc="left")
    ax.legend(frameon=False, loc="upper left")
    _ = xs
    save(fig, "fig8_time_vs_flops")


def fig_heatmap(summ):
    full = summ[summ.fraction == 1.0].set_index("model")
    ms = [m for m in ORDER if m in full.index]
    cols = [("accuracy_mean", "Accuracy", True), ("train_time_s_mean", "Train time", False),
            ("train_gpu_energy_kj_mean", "Train energy", False), ("train_peak_vram_torch_mb_mean", "Peak VRAM", False),
            ("infer_gpu_energy_j_mean", "Infer. energy", False), ("latency_b1_ms_mean", "Latency B=1", False)]
    M = np.array([[full.loc[m, c] for c, _, _ in cols] for m in ms], dtype=float)
    norm = M / M.max(0)
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ms)):
        for j, (c, lab, _) in enumerate(cols):
            v = M[i, j]
            txt = f"{v*100:.1f}%" if c == "accuracy_mean" else (f"{v:.0f}" if v >= 10 else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.3,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=6.3)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=6.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Multi-dimensional profile at n = 50k (shade = value / column max)", loc="left")
    save(fig, "fig9_profile_heatmap")


# --------------------------------------------------------------------------- #
def latex_tables(summ, comp, fits_t, fits_e, theo_mem, info):
    full = summ[summ.fraction == 1.0].set_index("model")
    ms = [m for m in ORDER if m in full.index]

    # Table: architecture complexity
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Params & MACs/img & Act./img & Layers \\", r"\midrule"]
    for m in ms:
        c = comp.loc[m]
        lines.append(f"{m} & {c.params/1e6:.2f}M & {c.macs_per_sample/1e6:.1f}M & "
                     f"{c.activations_per_sample/1e3:.0f}K & {int(c.weight_layers)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_complexity.tex"), "w").write("\n".join(lines))

    # Table: main results at full data
    has_cc = "train_total_energy_kj_mean" in full
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & Acc. (\%) & F1 (\%) & $T_{train}$ (s) & $E^{GPU}_{train}$ (kJ) & " +
             (r"$E^{tot}_{train}$ (kJ)" if has_cc else r"$P_{avg}$ (W)") + r" \\", r"\midrule"]
    for m in ms:
        r = full.loc[m]
        last = pm(r.train_total_energy_kj_mean, r.train_total_energy_kj_std) if has_cc else \
            pm(r.train_gpu_power_avg_w_mean, r.train_gpu_power_avg_w_std, "{:.1f}")
        lines.append(f"{m} & {pm(r.accuracy_mean*100, r.accuracy_std*100)} & {pm(r.f1_macro_mean*100, r.f1_macro_std*100)} & "
                     f"{pm(r.train_time_s_mean, r.train_time_s_std, '{:.1f}')} & "
                     f"{pm(r.train_gpu_energy_kj_mean, r.train_gpu_energy_kj_std)} & {last} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_main_train.tex"), "w").write("\n".join(lines))

    # Table: memory + inference
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & VRAM$_{tr}$ (MB) & RAM$_{tr}$ (MB) & $T_{inf}$ (ms) & $E_{inf}$ (J) & Lat.$_{B=1}$ (ms) \\",
             r"\midrule"]
    for m in ms:
        r = full.loc[m]
        lines.append(f"{m} & {pm(r.train_peak_vram_torch_mb_mean, r.train_peak_vram_torch_mb_std, '{:.0f}')} & "
                     f"{pm(r.train_peak_ram_mb_mean, r.train_peak_ram_mb_std, '{:.0f}')} & "
                     f"{pm(r.infer_time_s_mean*1000, r.infer_time_s_std*1000, '{:.0f}')} & "
                     f"{pm(r.infer_gpu_energy_j_mean, r.infer_gpu_energy_j_std, '{:.1f}')} & "
                     f"{pm(r.latency_b1_ms_mean, r.latency_b1_ms_std, '{:.2f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_memory_inference.tex"), "w").write("\n".join(lines))

    # Table: scaling exponents
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & $b_T$ [95\% CI] & $R^2_T$ & $b_E$ [95\% CI] & $R^2_E$ \\", r"\midrule"]
    for m in ms:
        t = fits_t[fits_t.model == m].iloc[0]
        e = fits_e[fits_e.model == m].iloc[0]
        lines.append(f"{m} & {t.exponent_b:.3f} [{t.ci95_low:.3f}, {t.ci95_high:.3f}] & {t.r2:.4f} & "
                     f"{e.exponent_b:.3f} [{e.ci95_low:.3f}, {e.ci95_high:.3f}] & {e.r2:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_scaling.tex"), "w").write("\n".join(lines))

    # Table: CPU vs GPU inference
    if "cpu_infer_ms_per_img_mean" in full:
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Model & GPU (ms/img) & CPU (ms/img) & Speed-up & CPU RAM (MB) \\", r"\midrule"]
        for m in ms:
            r = full.loc[m]
            g = r.infer_time_s_mean / 10000 * 1000
            c = r.cpu_infer_ms_per_img_mean
            lines.append(f"{m} & {g:.4f} & {pm(c, r.cpu_infer_ms_per_img_std, '{:.3f}')} & {c/g:.0f}$\\times$ & "
                         f"{r.cpu_infer_peak_ram_mb_mean:.0f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "tab_cpu_gpu.tex"), "w").write("\n".join(lines))


# --------------------------------------------------------------------------- #
SHORT = {"MLP": "Mlp", "SmallCNN": "Small", "MediumCNN": "Medium", "MobileNetV2": "Mobile", "ResNet18": "Res"}


def write_numbers(raw, summ, comp, fits_t, fits_e, info, report, theo_mem):
    """paper/numbers.tex: every number quoted in the paper text is a LaTeX macro generated from data."""
    full = summ[summ.fraction == 1.0].set_index("model")
    small = summ[summ.fraction == 0.1].set_index("model")
    L = ["% AUTO-GENERATED by src/analyze.py - do not edit by hand"]

    def cmd(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for m in ORDER:
        if m not in full.index:
            continue
        s, r = SHORT[m], full.loc[m]
        cmd(f"acc{s}", f"{r.accuracy_mean*100:.2f}")
        cmd(f"accsd{s}", f"{r.accuracy_std*100:.2f}")
        cmd(f"accsmall{s}", f"{small.loc[m, 'accuracy_mean']*100:.2f}")
        cmd(f"fone{s}", f"{r.f1_macro_mean*100:.2f}")
        cmd(f"auc{s}", f"{r.roc_auc_macro_mean*100:.2f}")
        cmd(f"ttrain{s}", f"{r.train_time_s_mean:.1f}")
        cmd(f"egpu{s}", f"{r.train_gpu_energy_kj_mean:.2f}")
        if "train_total_energy_kj_mean" in r:
            cmd(f"etot{s}", f"{r.train_total_energy_kj_mean:.2f}")
            cmd(f"wh{s}", f"{r.train_total_energy_kj_mean/3.6:.2f}")
        if "train_cc_emissions_g_mean" in r:
            cmd(f"co{s}", f"{r.train_cc_emissions_g_mean:.2f}")
        cmd(f"pavg{s}", f"{r.train_gpu_power_avg_w_mean:.1f}")
        cmd(f"util{s}", f"{r.train_gpu_util_avg_mean:.0f}")
        cmd(f"vram{s}", f"{r.train_peak_vram_torch_mb_mean:.0f}")
        cmd(f"vramnvml{s}", f"{r.train_peak_vram_nvml_mb_mean:.0f}")
        cmd(f"vramtheo{s}", f"{theo_mem.get(m, float('nan')):.0f}")
        cmd(f"vraminf{s}", f"{r.infer_peak_vram_torch_mb_mean:.0f}")
        cmd(f"ram{s}", f"{r.train_peak_ram_mb_mean:.0f}")
        cmd(f"tflops{s}", f"{r.achieved_tflops_mean:.2f}")
        cmd(f"tinf{s}", f"{r.infer_time_s_mean*1000:.0f}")
        cmd(f"einf{s}", f"{r.infer_gpu_energy_j_mean:.1f}")
        cmd(f"einfimg{s}", f"{r.infer_energy_mj_per_img_mean:.3f}")
        cmd(f"thr{s}", f"{r.infer_throughput_img_s_mean/1000:.1f}")
        cmd(f"lat{s}", f"{r.latency_b1_ms_mean:.2f}")
        if "cpu_infer_ms_per_img_mean" in r:
            cmd(f"cpums{s}", f"{r.cpu_infer_ms_per_img_mean:.3f}")
            cmd(f"speedup{s}", f"{r.cpu_infer_ms_per_img_mean / (r.infer_time_s_mean / 10):.0f}")
        cmd(f"params{s}", f"{comp.loc[m, 'params']/1e6:.2f}")
        cmd(f"macs{s}", f"{comp.loc[m, 'macs_per_sample']/1e6:.1f}")
        cmd(f"acts{s}", f"{comp.loc[m, 'activations_per_sample']/1e3:.0f}")
        t = fits_t[fits_t.model == m]
        e = fits_e[fits_e.model == m]
        if not t.empty:
            cmd(f"bt{s}", f"{t.iloc[0].exponent_b:.3f}")
            cmd(f"be{s}", f"{e.iloc[0].exponent_b:.3f}")
            cmd(f"rt{s}", f"{t.iloc[0].r2:.4f}")
        cmd(f"tpse{s}", f"{r.time_per_sample_epoch_us_mean:.1f}")
        cmd(f"epse{s}", f"{r.energy_per_sample_epoch_mj_mean:.3f}")
    cmd("idlepower", f"{info.get('gpu_idle_power_w', 0):.1f}")
    cmd("nruns", f"{len(raw)}")
    cmd("totalhours", f"{raw['train_time_s'].sum()/3600:.2f}")
    if "train_cc_total_energy_j" in raw:
        cmd("totalwh", f"{raw['train_cc_total_energy_j'].sum()/3600:.1f}")
        cmd("totalco", f"{raw['train_cc_emissions_g'].sum():.1f}")
    cmd("totalgpuwh", f"{raw['train_gpu_energy_j'].sum()/3600:.1f}")
    cmd("rhoflops", f"{report['spearman_flops_time'][0]:.2f}")
    cmd("rhoparams", f"{report['spearman_params_time'][0]:.2f}")
    cmd("rlogflops", f"{report['pearson_logflops_logtime'][0]:.3f}")
    cmd("rvram", f"{report['pearson_theo_vs_meas_vram'][0]:.3f}")
    if report.get("kj_per_extra_acc_point") is not None:
        cmd("dacc", f"{report['resnet_vs_medium_delta_acc_pts']:.2f}")
        cmd("denergy", f"{report['resnet_vs_medium_delta_energy_kj']:.2f}")
        cmd("kjperpoint", f"{report['kj_per_extra_acc_point']:.2f}")
    # CPU/GPU energy split from CodeCarbon (full data, all models)
    fr = raw[raw.fraction == 1.0]
    if "train_cc_cpu_energy_j" in fr:
        tot = fr[["train_cc_cpu_energy_j", "train_cc_gpu_energy_j", "train_cc_ram_energy_j"]].sum()
        cmd("sharegpu", f"{tot.train_cc_gpu_energy_j / tot.sum() * 100:.0f}")
        cmd("sharecpu", f"{tot.train_cc_cpu_energy_j / tot.sum() * 100:.0f}")
        cmd("shareram", f"{tot.train_cc_ram_energy_j / tot.sum() * 100:.0f}")
        cmd("ccvsnvml", f"{(fr.train_cc_gpu_energy_j / fr.train_gpu_energy_j).mean():.3f}")
    # coefficient of variation of training time / energy (run-to-run noise)
    cv_t = (summ["train_time_s_std"] / summ["train_time_s_mean"]).median() * 100
    cv_e = (summ["train_gpu_energy_j_std"] / summ["train_gpu_energy_j_mean"]).median() * 100
    cmd("cvtime", f"{cv_t:.1f}")
    cmd("cvenergy", f"{cv_e:.1f}")
    open(os.path.join(PAPER, "numbers.tex"), "w").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main():
    raw, comp, info = load()
    summ = summarize(raw)
    summ.to_csv(os.path.join(RES, "summary_by_model_fraction.csv"), index=False)

    fits_t = scaling_fit(raw, "train_time_s")
    fits_e = scaling_fit(raw, "train_gpu_energy_j")
    fits_vram = scaling_fit(raw, "train_peak_vram_torch_mb")
    fits = pd.concat([fits_t, fits_e, fits_vram])
    fits.to_csv(os.path.join(RES, "scaling_exponents.csv"), index=False)

    batch = int(raw["batch_size"].iloc[0])
    fig_scaling(raw, summ, fits_t, "train_time_s", "Training time (s, log scale)", "fig1_train_time_scaling",
                "Training time vs. data size (fit $T=c\\,n^b$)")
    fig_scaling(raw, summ, fits_e, "train_gpu_energy_kj", "Training GPU energy (kJ, log scale)",
                "fig2_train_energy_scaling", "Training energy vs. data size (fit $E=c\\,n^b$)")
    fig_accuracy(summ)
    front = fig_tradeoff(summ, comp)
    theo_mem = fig_memory(summ, comp, batch)
    fig_efficiency(summ, info)
    fig_inference(summ)
    fig_time_vs_flops(summ)
    fig_heatmap(summ)
    latex_tables(summ, comp, fits_t, fits_e, theo_mem, info)

    # cross-model correlation between theory and measurement (full data)
    full = summ[summ.fraction == 1.0]
    rho_t = stats.spearmanr(full["train_tflop_mean"], full["train_time_s_mean"])
    rho_p = stats.spearmanr(full["params_mean"], full["train_time_s_mean"])
    pear_t = stats.pearsonr(np.log(full["train_tflop_mean"]), np.log(full["train_time_s_mean"]))
    mem_meas = full.set_index("model")["train_peak_vram_torch_mb_mean"]
    mem_theo = pd.Series(theo_mem)
    pear_m = stats.pearsonr(mem_theo[mem_meas.index], mem_meas)

    # pooled statistical test: is accuracy of ResNet18 > MediumCNN etc. (Welch t-test on full data)
    tests = []
    fr = raw[raw.fraction == 1.0]
    for a, b in [("ResNet18", "MobileNetV2"), ("ResNet18", "MediumCNN"), ("MobileNetV2", "MediumCNN")]:
        if a in fr.model.values and b in fr.model.values:
            for col in ("accuracy", "train_gpu_energy_j"):
                t = stats.ttest_ind(fr[fr.model == a][col], fr[fr.model == b][col], equal_var=False)
                tests.append({"a": a, "b": b, "metric": col, "t": t.statistic, "p": t.pvalue})
    pd.DataFrame(tests).to_csv(os.path.join(RES, "welch_tests.csv"), index=False)

    # energy-efficiency indicators
    eff = full[["model", "accuracy_mean", "train_gpu_energy_kj_mean", "train_time_s_mean",
                "infer_gpu_energy_j_mean"]].copy()
    if "train_cc_emissions_g_mean" in full:
        eff["train_co2_g"] = full["train_cc_emissions_g_mean"]
    eff["acc_points_per_kj"] = eff["accuracy_mean"] * 100 / eff["train_gpu_energy_kj_mean"]
    base = eff.set_index("model")
    if "MediumCNN" in base.index and "ResNet18" in base.index:
        d_acc = (base.loc["ResNet18", "accuracy_mean"] - base.loc["MediumCNN", "accuracy_mean"]) * 100
        d_e = base.loc["ResNet18", "train_gpu_energy_kj_mean"] - base.loc["MediumCNN", "train_gpu_energy_kj_mean"]
        eff_note = {"resnet_vs_medium_delta_acc_pts": d_acc, "resnet_vs_medium_delta_energy_kj": d_e,
                    "kj_per_extra_acc_point": d_e / d_acc if d_acc else None}
    else:
        eff_note = {}
    eff.to_csv(os.path.join(RES, "efficiency_indicators.csv"), index=False)

    report = {
        "pareto_front": front,
        "spearman_flops_time": [rho_t.statistic, rho_t.pvalue],
        "spearman_params_time": [rho_p.statistic, rho_p.pvalue],
        "pearson_logflops_logtime": [pear_t.statistic, pear_t.pvalue],
        "pearson_theo_vs_meas_vram": [pear_m.statistic, pear_m.pvalue],
        "theoretical_vram_mb": theo_mem,
        **eff_note,
    }
    with open(os.path.join(RES, "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    write_numbers(raw, summ, comp, fits_t, fits_e, info, report, theo_mem)

    pd.set_option("display.width", 200)
    print(fits.to_string(index=False))
    print(json.dumps(report, indent=2, default=float))
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()
