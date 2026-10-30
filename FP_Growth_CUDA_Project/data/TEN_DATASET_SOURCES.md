# Exactly 10 offline benchmark datasets

All files here are transaction baskets (one whitespace-separated itemset per line), with **no class target labels**. The ten default benchmarks below include **four UCI-origin datasets and six seeded synthetic datasets**. Synthetic data is not represented as UCI-origin. Files with other names (toy/repeated digits) are supplied for correctness/scaling but are not in the ten-benchmark catalog.

1. **UCI Iris (3-bin)** — `uci_iris_150.txt`; 150 transactions; 25% minimum support; UCI-derived (scikit-learn load_iris). 150 samples; four continuous features quantile-discretized into three equal-frequency categories; target omitted
2. **UCI Wine (median-bin)** — `uci_wine_178.txt`; 178 transactions; 40% minimum support; UCI-derived (scikit-learn load_wine). 178 samples; 13 real-valued features median-binarized; target omitted
3. **UCI WDBC (median-bin)** — `uci_wdbc_569.txt`; 569 transactions; 40% minimum support; UCI-derived (scikit-learn load_breast_cancer). 569 samples; 30 features median-binarized; target omitted
4. **UCI Digits (nonzero)** — `uci_digits_1797.txt`; 1,797 transactions; 80% minimum support; UCI-origin handwritten digits test set (scikit-learn load_digits). 1797 samples; nonzero pixels become basket items; digit target omitted
5. **Synthetic grocery** — `synthetic_grocery_3000.txt`; 3,000 transactions; 12% minimum support; Synthetic (seed 20261029 + 101). 3000 grocery baskets with repeated common staple items and occasional thematic group
6. **Synthetic correlated bundles** — `synthetic_bundles_6000.txt`; 6,000 transactions; 15% minimum support; Synthetic (seed 20261029 + 202). 6000 baskets with planted clusters and independent background items
7. **Synthetic sparse clickstream** — `synthetic_sparse_12000.txt`; 12,000 transactions; 2% minimum support; Synthetic (seed 20261029 + 303). 12000 baskets with 2-6 selected items from a 300-item long-tail vocabulary
8. **Synthetic dense sensors** — `synthetic_dense_sensors_8000.txt`; 8,000 transactions; 36% minimum support; Synthetic (seed 20261029 + 404). 8000 simulated dense sensor-state baskets with 32 binary items
9. **Synthetic long-tail catalog** — `synthetic_longtail_20000.txt`; 20,000 transactions; 5% minimum support; Synthetic (seed 20261029 + 505). 20000 baskets with highly skewed frequency distribution over 150 catalog entries
10. **Synthetic retail 50k** — `synthetic_retail_50000.txt`; 50,000 transactions; 5% minimum support; Synthetic (seed 3582026). 50000 reproducible skewed retail baskets; same set as the previous release

## Original dataset references

- Iris (UCI #53): https://archive.ics.uci.edu/dataset/53/iris
- Wine (UCI #109): https://archive.ics.uci.edu/dataset/109/wine
- Breast Cancer Wisconsin Diagnostic (UCI #17): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Optical Recognition of Handwritten Digits (UCI #80): https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Offline scikit-learn copies: https://scikit-learn.org/stable/datasets/toy_dataset.html

**Preprocessing matters:** Iris uses feature-wise terciles; Wine and WDBC use per-feature median bins; Digits maps positive pixel intensities to pixel-present tokens. The feature transformation and omission of labels change the itemset interpretation.

The three optional extra original UCI categorical datasets Mushroom, Chess, Connect-4 can be downloaded using `python3 scripts/get_uci.py all` on a network-enabled computer, but they are **not part of the ten offline benchmark inputs** and have **not been benchmarked**.
