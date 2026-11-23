# Empirical Time–Space–Energy–Performance Complexity of Deep Learning Models on CIFAR-10

BYM303 Algorithm Analysis — Course Project (2026–2027 Fall)

| Student No | Name |
|---|---|
| 240609043 | Baran Kemal Kılıç |
| 240609009 | İrem Serra Onmaz |
| 240609050 | Nihat Güney |
| 240609014 | Süleyman Canberk Eğriboz |

Five image classifiers (MLP, SmallCNN, MediumCNN, MobileNetV2, ResNet-18) are trained from
scratch on CIFAR-10 with 4 training-set sizes (5k, 12.5k, 25k, 50k) and 5 random seeds
(**100 runs**). For every run we measure, separately for training and inference:

* **Time** – wall-clock training time, batched GPU inference time, batch-1 latency, CPU inference time
* **Space** – peak process RAM (psutil), peak VRAM (PyTorch allocator + NVML)
* **Energy** – GPU energy from the NVML hardware energy counter, CPU+GPU+RAM energy and CO₂ from CodeCarbon (Turkish grid intensity)
* **Performance** – Accuracy, macro Precision / Recall / F1, macro ROC-AUC

and compare them with the analytical complexity of each network (parameters *P*, MACs per image *F*,
activations per image *A*): training cost Θ(E·n·F), inference Θ(m·F), memory Θ(P + B·A).

## Repository layout

```
src/
  models.py               five architectures
  complexity.py           analytical P / MACs / activations via forward hooks
  data.py                 CIFAR-10 loading, stratified subsets, GPU augmentation
  measure.py              time, RAM, VRAM, NVML energy, CodeCarbon wrapper
  run_experiments.py      experiment driver (resumable, writes results/raw_runs.csv)
  analyze.py              statistics, scaling fits, figures, LaTeX tables
  bench_precision_layout.py  fp16 vs bf16 / NCHW vs channels_last micro-benchmark
results/
  raw_runs.csv            one row per run (100 rows, ~60 columns)
  summary_by_model_fraction.csv   mean and std per (model, n)
  scaling_exponents.csv   fitted exponent b of T = c n^b, E = c n^b (95% CI)
  model_complexity.csv    analytical P, MACs, activations
  system_info.json        hardware / software versions, idle GPU power
  analysis_report.json    correlations theory vs measurement, Pareto front
  welch_tests.csv         Welch t-tests between models
figures/                  all figures (PDF for the paper, PNG for slides)
paper/                    IEEE conference paper (LaTeX, Overleaf-ready)
```

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src\run_experiments.py --quick   # 1-minute smoke test
.venv\Scripts\python src\run_experiments.py           # full protocol (~1-1.5 h on RTX 4060 Laptop)
.venv\Scripts\python src\analyze.py                   # figures + tables
```

The driver is resumable: if it is interrupted, run it again and it continues where it stopped.
Keep the laptop plugged in and on the *High performance* power plan while measuring.
