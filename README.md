# Congestion-State Classification: LightGBM vs MLP on Metro Interstate Traffic Volume

Motivation: "Fully-autonomous, vision-based traffic signal control: From simulation to reality" (2022) proposes a deep reinforcement learning controller that acts on live RGB camera footage of an intersection, so the quality of the learned state-input model matters for real-world traffic control. This experiment asks a simpler, tabular version of that question: can an interpretable, class-balanced gradient-boosted model classify high-congestion states better than a standard neural network baseline?

Question: Can a class-balanced, CV-tuned LightGBM model with an interpretable feature set beat a standard MLP at classifying high-congestion intersection states from traffic-context features?

Dataset: UCI Metro Interstate Traffic Volume (hourly traffic counts with weather and holiday fields), fetched with the `ucimlrepo` package. The target is a binary high-congestion label (volume above the training median). This is not the paper's own dataset; it is a tabular stand-in for the paper's camera-based signal state.

Baseline vs improved:
- Baseline: standard scikit-learn MLPClassifier on standardised and one-hot encoded features.
- Improved: class-balanced LightGBM with hyper-parameters chosen by time-series cross-validation on the training portion, using identical features and preprocessing.

Evaluation: chronological 80/20 split (no future leakage), main metric macro-F1 on the held-out test set, with accuracy and log-loss as secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/macro_f1_comparison.png`).

## Reference

This project is motivated by the following paper:

> D Garg, M Chli, G Vogiatzis (2022). *Fully-autonomous, vision-based traffic signal control: From simulation to reality*. http://publications.aston.ac.uk/id/eprint/43543/

Paper author: Prof. George Vogiatzis, Aston University (g.vogiatzis@aston.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
