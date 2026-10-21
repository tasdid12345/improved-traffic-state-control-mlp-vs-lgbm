# Baseline MLP vs. class-balanced LightGBM on UCI Metro Interstate Traffic Volume

Question: can a class-balanced, CV-tuned LightGBM with an interpretable feature set beat a standard MLP at classifying high-congestion states (traffic volume above the training median) on the UCI Metro Interstate Traffic Volume data?

Method:
- Data fetched with `ucimlrepo` (dataset id 492), capped at 200k rows, sorted by time.
- Chronological 80/20 split; the threshold is the median of the training portion only.
- Baseline: scikit-learn `MLPClassifier` (64-32, early stopping) on the base features.
- Improved: class-balanced LightGBM on the base features plus engineered time-of-week, cyclical hour, rush-hour, night, clear-sky and precipitation features. Hyper-parameters are chosen by 3-fold `TimeSeriesSplit` CV on the training portion only.
- The test set is used once for the final evaluation, over seeds 42, 43 and 44.

Run: `python analysis.py` (writes `results.json` and `figures/macro_f1_comparison.png`).

## Results (latest run)

On the UCI Metro Interstate Traffic Volume dataset, a standard MLP classifier reached 0.950 macro-F1 on the held-out test set, while my class-balanced, CV-tuned LightGBM reached 0.951, a negligible gain within seed variation (std 0.001).

| Metric | Value |
|---|---|
| baseline_score | 0.9502 |
| improved_score | 0.9508 |
| improvement | 0.0005871 |
| baseline_score_std | 0.001327 |
| improved_score_std | 0.00008431 |
| baseline_accuracy | 0.9502 |
| improved_accuracy | 0.9508 |
| baseline_logloss | 0.1663 |
| improved_logloss | 0.1478 |
| lgbm_best_cv_macro_f1 | 0.9563 |
| n_seeds | 3 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Selected LightGBM params {'learning_rate': 0.05, 'min_child_samples': 300, 'n_estimators': 200, 'num_leaves': 7} by time-series CV on training data only. Improved model uses extra engineered features that the baseline does not receive; baseline features and MLP are unchanged. Label is a tabular stand-in for the paper's camera-based signal state, not its RGB footage. Chronological split means the test period is later than training (distribution shift). MLP is untuned by design (fixed architecture), so tuning budget is not equal. Seeds vary model randomness only, not the split. Class-balanced LightGBM probabilities are not calibrated to the natural class prior, which can affect log-loss. Total runtime 54s.

## Reference

This project is motivated by the following paper:

> D Garg, M Chli, G Vogiatzis (2022). *Fully-autonomous, vision-based traffic signal control: From simulation to reality*. http://publications.aston.ac.uk/id/eprint/43543/

Paper author: Prof. George Vogiatzis, Aston University (g.vogiatzis@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
