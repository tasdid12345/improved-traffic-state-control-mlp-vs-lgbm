# Baseline MLP vs. class-balanced LightGBM on UCI Metro Interstate Traffic Volume

Question: can a class-balanced, CV-tuned LightGBM with an interpretable feature set beat a standard MLP at classifying high-congestion states (traffic volume above the training median) on the UCI Metro Interstate Traffic Volume data?

Method:
- Data fetched with `ucimlrepo` (dataset id 492), capped at 200k rows, sorted by time.
- Chronological 80/20 split; the threshold is the median of the training portion only.
- Baseline: scikit-learn `MLPClassifier` (64-32, early stopping) on the base features.
- Improved: class-balanced LightGBM on the base features plus engineered time-of-week, cyclical hour, rush-hour, night, clear-sky and precipitation features. Hyper-parameters are chosen by 3-fold `TimeSeriesSplit` CV on the training portion only.
- The test set is used once for the final evaluation, over seeds 42, 43 and 44.

Run: `python analysis.py` (writes `results.json` and `figures/macro_f1_comparison.png`).

## Reference

This project is motivated by the following paper:

> D Garg, M Chli, G Vogiatzis (2022). *Fully-autonomous, vision-based traffic signal control: From simulation to reality*. http://publications.aston.ac.uk/id/eprint/43543/

Paper author: Prof. George Vogiatzis, Aston University (g.vogiatzis@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
