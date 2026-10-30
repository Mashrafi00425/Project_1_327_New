# Reference comparison: what changed and what remains

The supplied `ESE_327_proj1-main.zip` was examined as a **presentation and benchmarking reference**, not used as a source of performance numbers. That archive contains a README with extensive algorithm explanations and results from 13 UCI datasets, plus CPU/GPU CSV records and per-dataset four-panel PNG figures. **It does not contain a finished PDF report**.

## Improvements made to this independently implemented CUDA FP-Growth project

1. Added a four-level minimum-support sweep for **each of our ten existing datasets**: 40 benchmark settings in total.
2. Added `scripts/benchmark_sweep.py` to repeat CPU measurements and, when run in the NVIDIA lab, compare CPU and actual CUDA timings on the **same machine**. It records process RAM, CUDA buffer high-water, counts, tree metrics, GPU device name and speedup. It saves incremental CSV/JSON results as it runs.
3. Added a 16-page printable report, `report/FP_Growth_Expanded_10_Dataset_Report.pdf`, with a **four-panel figure for each dataset**: algorithm runtime, frequent-itemset count, memory, CPU/GPU speedup. Each figure includes the exact values in a per-dataset table.
4. Rewired `python3 scripts/generate_report.py` to generate this expanded PDF and to refresh the original report PDF filename too.
5. Preserved the original C++17 FP-Growth code and our own CUDA counting kernel. **Our CUDA architecture remains a CPU/GPU hybrid**; we do not claim GPU-resident FP-tree construction/mining.
6. Avoided copying any CPU or NVIDIA timings from the classmate's results. CUDA-speedup panels are labeled PENDING until `--with-cuda` produces real device measurements.

## Differences that still matter

- The reference includes **13 authentic UCI datasets**; our ten-file default still contains **4 UCI-derived and 6 synthetic** datasets. The assignment specifically calls for UCI benchmark examples, so this collection may need to be replaced with ten original UCI datasets to meet the instructor's expectations. Do not label our synthetic datasets as UCI.
- The reference uses a largely device-based tree and mining algorithm; ours accelerates weighted support histograms at all recursive levels while leaving FP-tree construction/mining recursion on CPU. It is genuine CUDA work, but **not the same architecture or expected performance**.
- The classmate's RTX 3060 Ti and Jetson results belong to their implementation and hardware, **not this project's performance measurements**.
- Previous lab runs of `benchmark_cuda.py` produce 10 one-threshold CPU/GPU rows. The **new** support sweep requires another lab run for 40 CPU/GPU rows; it does not manufacture the missing 30 GPU observations from old results.

## Lab commands

```bash
make
make cuda
python3 scripts/verify_cuda.py --random-cases 20 | tee results/real_gpu_verification.log
python3 scripts/benchmark_sweep.py --with-cuda --repeats 3
python3 scripts/generate_report.py
```

The updated PDF is at `report/FP_Growth_CUDA_10_Dataset_Report.pdf` (and the extended named copy). If Python errors on missing packages:

```bash
sudo apt install python3-reportlab python3-matplotlib
```

A Python virtual environment with those packages also works. **On the lab machine, preserve any existing `results/cuda_benchmark_results.csv` and `results/cuda_benchmark_results.json` before replacing this folder**: they hold your earlier actual GPU measurements and are not included in this packaged revision.
