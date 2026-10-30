# UPDATED: reference-style support-threshold study (October 29, 2026)

**Start here:** Read `REPORT_REFERENCE_COMPARISON.md` for exactly what was learned from the classmate's reference and which improvements were implemented without copying their code or results. The new report has **40 support-threshold observations (4 per dataset), 10 four-panel charts and per-dataset tables**, rather than only one setting per dataset.

The expanded PDF is `report/FP_Growth_CUDA_10_Dataset_Report.pdf` (also saved as `report/FP_Growth_Expanded_10_Dataset_Report.pdf`). In this ZIP it contains actual measured CPU data, with CUDA panels clearly marked **PENDING**. To fill them using your own NVIDIA lab measurements, run:

```bash
make
make cuda
python3 scripts/verify_cuda.py --random-cases 20 | tee results/real_gpu_verification.log
python3 scripts/benchmark_sweep.py --with-cuda --repeats 3
python3 scripts/generate_report.py
```

**Do not submit the pre-lab PDF as proof of GPU speedups.** Its CPU results were measured in a preparation environment, not your lab. The reference project has 13 UCI datasets; ours still has **four UCI-derived plus six synthetic**. Verify with your instructor whether all ten must be UCI examples. For missing Python dependencies use `sudo apt install python3-reportlab python3-matplotlib` on Ubuntu if permitted.

### Quick CPU-only regeneration (no CUDA machine required)

```bash
python3 scripts/benchmark_sweep.py --repeats 3
python3 scripts/generate_report.py
```

---

# CUDA FP-Growth - TEN-dataset version (October 29, 2026)

**This is the updated version with ten distinct OFFLINE benchmark datasets**. The benchmark catalog contains **four UCI-derived datasets** and **six independent seeded synthetic datasets**. The five-basket toy file is an additional manual correctness example (not one of the ten benchmarks). Their exact names, support thresholds and SHA-256 hashes appear in `scripts/benchmark_catalog.json` and `data/TEN_DATASET_SOURCES.md`.

The algorithm is recursive FP-Growth (FP-tree + header table + weighted conditional pattern bases), implemented in C++17. In the CUDA build, weighted item-support histograms run on the NVIDIA GPU **for both the root and all recursive conditional FP-trees**. Building/traversing trees, recursion, and CSV/JSON output remain on the CPU. This is an honest **CUDA-accelerated HYBRID**, not a wholly GPU-resident algorithm.

## TOMORROW'S LAB COMMANDS - test actual NVIDIA CUDA

Open a terminal in this folder (where `Makefile` is located), then:

```bash
nvcc --version
nvidia-smi                          # skip on Jetson if unavailable
make                               # builds ./fp_growth CPU baseline
make cuda                          # builds ./fp_growth_cuda with nvcc
./fp_growth_cuda data/toy_transactions.txt 3 --output results/toy_gpu.csv --summary results/toy_gpu.json
python3 scripts/verify_cuda.py --random-cases 20
python3 scripts/benchmark_cuda.py --repeats 3
python3 scripts/generate_report.py
```

**Expected result:** toy file -> 8 frequent itemsets, 5 CUDA histogram kernel calls, and a real NVIDIA `cuda_device` name in the JSON. The strict verifier compares **all itemsets and their support values**, not just totals, across **all ten benchmark datasets**, toy example, and 20 random test cases. The benchmark produces `results/cuda_benchmark_results.csv` and `.json` for exactly 10 datasets. The PDF generator will add the GPU results only after it sees a complete ten-row real-device CSV. Reprint the final PDF after that command.

**Important:** Here, without `nvcc` or NVIDIA GPU, we compiled and ran the CPU code, ran all seven existing unit-test groups, and compared the ten benchmarks + toy + 20 random cases against a CPU simulator of the CUDA API. The **actual `.cu` device code has NOT been compiled or run**; no GPU times were invented. GPU verification in your lab is required before submission.

If `nvcc` is installed but not on PATH:

```bash
/usr/local/cuda/bin/nvcc --version
make cuda NVCC=/usr/local/cuda/bin/nvcc
```

For an Orin Nano with an appropriate CUDA toolkit, `make cuda NVCCFLAGS='-std=c++17 -O2 -arch=sm_87'` may work. Do not use `sm_87` for a different NVIDIA GPU without checking the GPU compute capability.

## Exactly 10 benchmark datasets (included offline)

| No. | Dataset | File | Transactions | Min support | Origin |
|--:|---|---|---:|---|---|
| 1 | Iris (3-bin features) | `uci_iris_150.txt` | 150 | 25% | UCI origin via scikit-learn |
| 2 | Wine (median-bin features) | `uci_wine_178.txt` | 178 | 40% | UCI origin via scikit-learn |
| 3 | Wisconsin Diagnostic Breast Cancer | `uci_wdbc_569.txt` | 569 | 40% | UCI origin via scikit-learn |
| 4 | Handwritten Digits (pixel-nonzero) | `uci_digits_1797.txt` | 1,797 | 80% | UCI origin via scikit-learn |
| 5 | Grocery baskets | `synthetic_grocery_3000.txt` | 3,000 | 12% | Independently generated synthetic |
| 6 | Correlated bundles | `synthetic_bundles_6000.txt` | 6,000 | 15% | Independently generated synthetic |
| 7 | Sparse clickstream | `synthetic_sparse_12000.txt` | 12,000 | 2% | Independently generated synthetic |
| 8 | Dense sensor states | `synthetic_dense_sensors_8000.txt` | 8,000 | 36% | Independently generated synthetic |
| 9 | Long-tail catalog | `synthetic_longtail_20000.txt` | 20,000 | 5% | Independently generated synthetic |
| 10 | Retail 50,000 | `synthetic_retail_50000.txt` | 50,000 | 5% | Independently generated synthetic |

The files are not copies of one another or merely rescaled versions; they represent distinct feature distributions and row sizes. The six synthetic files are **not claimed to come from UCI**. The four UCI-origin data samples are transformed into transactions for association mining, and classification labels are excluded. Details and original UCI links are in `data/TEN_DATASET_SOURCES.md`.

The assignment requests UCI examples. Four are provided and benchmarked offline. The optional `scripts/get_uci.py` downloads original UCI Mushroom, Chess, and Connect-4 (requires network); they are supplementary, **not among the ten offline benchmarks** and no measurements on them are claimed. If the instructor requires **ten UCI-origin** datasets specifically, this four-UCI/six-synthetic version would need extra sourcing and experiments before submission.

## Main files

| Path | Purpose |
|---|---|
| `src/fp_growth.cpp` | Real FP-Growth, supports CPU or CUDA host compilation |
| `src/cuda_histogram.cu` | Actual CUDA 64-bit weighted support-count kernel |
| `src/cuda_histogram.hpp` | Shared GPU histogram header / counters |
| `tests/test_fp_growth.py` | Seven CPU tests including exhaustive oracle for 100 random inputs |
| `tests/cuda_histogram_emulator.cpp` | Non-GPU test double for interface validation only |
| `scripts/verify_cuda.py` | Exact CPU-vs-actual-GPU pattern checking for ten datasets |
| `scripts/benchmark.py` | CPU runtime / RAM 10-dataset benchmarks |
| `scripts/benchmark_cuda.py` | Lab GPU vs same-lab-CPU 10-dataset runtime / RAM / kernel benchmarks |
| `scripts/benchmark_catalog.json` | Single authoritative list of the TEN experiments |
| `scripts/prepare_ten_datasets.py` | Regenerate the same datasets, needs NumPy/scikit-learn |
| `scripts/generate_report.py` | Generates nine-page printable PDF, needs reportlab/matplotlib |
| `report/FP_Growth_CUDA_10_Dataset_Report.pdf` | Nine-page report with 10 real CPU results and GPU worksheet |
| `results/benchmark_results.csv` | Measured ten-dataset CPU results (preparation machine) |

### Manual use and CPU tests

```bash
make
make test
./fp_growth data/uci_iris_150.txt 25% --output results/iris_patterns.csv --summary results/iris_summary.json
./fp_growth data/synthetic_retail_50000.txt 5% --no-output --summary results/retail_summary.json
python3 scripts/benchmark.py --repeats 5
```

### Optional simulator - NOT an NVIDIA GPU test

```bash
make cuda_sim
python3 scripts/verify_cuda.py --cuda-bin fp_growth_cuda_sim --random-cases 20
```

The simulator runs the counting phase on CPU via a mock interface. It verifies the conditional path orchestration, but **cannot prove CUDA kernel correctness or compilation**.

## Outputs and minimum support

Each transaction file contains one whitespace-delimited set of items per line. Duplicate items in the same row are counted once. `--output` writes every frequent itemset and its exact integer support to CSV; `--no-output` suppresses potentially large pattern output, but the algorithm still mines the patterns. `--summary` writes JSON statistics including transaction count, support threshold, itemset count, runtime and peak RSS, and (for CUDA) kernel counts, device name and high-water CUDA buffer allocation.

An integer minimum support is an absolute number of transactions; a decimal such as `0.25` is fractional, and `25%` is percentage support. Dense inputs with low support thresholds may produce exponentially many outputs.

## How to finalize the submitted ZIP after lab testing

1. Run `make cuda`, the exact verifier and all ten benchmarks on the same GPU-enabled lab computer. Fix any mismatches before proceeding.
2. Inspect `results/cuda_benchmark_results.csv` and confirm there are **ten rows** and the GPU device is real.
3. Run `python3 scripts/generate_report.py` to populate the lab table, then read `report/FP_Growth_CUDA_10_Dataset_Report.pdf` carefully.
4. ZIP **this project folder** with the updated PDF, code, datasets and CSV. You may omit built binary executables. Print the *updated* PDF and email the final ZIP per the actual instructor instructions.

**No fabricated measurements:** reported CPU values are actual local benchmark results; GPU numbers will be present only after actual lab execution. Performance may be worse than CPU on small trees due to kernel-launch and transfer costs.

## References

- Han, Pei & Yin, *Mining Frequent Patterns without Candidate Generation* (2000): https://doi.org/10.1145/342009.335372
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
