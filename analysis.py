"""Baseline MLP vs class-balanced, CV-tuned LightGBM with extra engineered features on UCI Metro Interstate Traffic Volume.
Target: high-congestion label (traffic_volume > training-set median). Chronological 80/20 split."""
import json, os, socket, sys, time
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

SEED, SEEDS, MAX_ROWS, DATASET_ID = 42, [42, 43, 44], 200_000, 492
SOURCE_URL = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
NUM = ["hour", "dayofweek", "month", "is_weekend", "temp", "clouds_all", "rain_1h", "snow_1h"]
# Extra engineered numeric features, used ONLY by the improved LightGBM model.
EXTRA_NUM = ["hour_of_week", "hour_sin", "hour_cos", "is_rush", "is_night", "clear_sky", "is_precip"]
NUM_IMP = NUM + EXTRA_NUM
CAT = ["holiday", "weather_main"]
socket.setdefaulttimeout(60)  # 60 s timeout for the underlying HTTP calls


def load_data():
    """Fetch the dataset via ucimlrepo, retrying with exponential backoff on 429/5xx/network errors."""
    for attempt in range(5):
        try:
            ds = fetch_ucirepo(id=DATASET_ID)
            return ds.data.features.copy(), ds.data.targets["traffic_volume"].copy()
        except Exception as e:  # ucimlrepo surfaces HTTP errors as exceptions
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status != 429 and status < 500:
                raise  # non-retryable client error
            wait = 2 ** attempt
            print(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("could not download dataset after retries")


def build_features(df):
    """Shared base feature set (interpretable, no raw timestamp, no leakage of volume). Used by the baseline."""
    dt = pd.to_datetime(df["date_time"])
    return pd.DataFrame({
        "hour": dt.dt.hour, "dayofweek": dt.dt.dayofweek, "month": dt.dt.month,
        "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
        "temp": df["temp"], "clouds_all": df["clouds_all"],
        "rain_1h": np.log1p(df["rain_1h"]), "snow_1h": np.log1p(df["snow_1h"]),  # heavy-tailed -> log
        "holiday": df["holiday"].fillna("None").astype(str),
        "weather_main": df["weather_main"].fillna("Unknown").astype(str),
    })


def build_improved_features(feats):
    """Base features plus engineered time-of-week / rush-hour / weather interaction features.
    Computed from the base feature frame only (no target information)."""
    out = feats.copy()
    out["hour_of_week"] = feats["dayofweek"] * 24 + feats["hour"]
    out["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24.0)
    out["is_rush"] = (feats["hour"].between(6, 9) | feats["hour"].between(15, 18)).astype(int) \
        * (1 - feats["is_weekend"])
    out["is_night"] = feats["hour"].isin(range(0, 6)).astype(int)
    out["clear_sky"] = (feats["clouds_all"] < 20).astype(int)
    out["is_precip"] = ((feats["rain_1h"] > 0) | (feats["snow_1h"] > 0)).astype(int)
    return out


def make_pre(num=NUM):
    """Shared preprocessing: standardise numeric, one-hot categorical (fitted on training folds only)."""
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def make_mlp(seed):
    """BASELINE: standard scikit-learn MLP with a fixed, untuned architecture. Unchanged."""
    return make_pipeline(make_pre(NUM), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                                      early_stopping=True, random_state=seed))


def make_lgbm(params, seed):
    """IMPROVED: class-balanced LightGBM with engineered features and CV-selected hyper-parameters."""
    return make_pipeline(make_pre(NUM_IMP), LGBMClassifier(
        class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1, n_jobs=2, **params))


def main():
    t0 = time.time()
    X, y_raw = load_data()
    df = X.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["vol"] = y_raw.to_numpy()
    if len(df) > MAX_ROWS:  # runtime cap for very large data
        df = df.sample(MAX_ROWS, random_state=SEED)
    df = df.sort_values("date_time").reset_index(drop=True)  # chronological order

    # Chronological split: earliest 80% train, latest 20% test (no future leakage)
    feats = build_features(df)
    cut = int(0.8 * len(df))
    thr = df["vol"].iloc[:cut].median()  # threshold from training portion only
    y = (df["vol"] > thr).astype(int).to_numpy()
    Ftr, Fte, ytr, yte = feats.iloc[:cut], feats.iloc[cut:], y[:cut], y[cut:]
    Itr, Ite = build_improved_features(Ftr), build_improved_features(Fte)
    print(f"rows={len(df)} train={len(Ftr)} test={len(Fte)} threshold={thr} "
          f"test positive rate={yte.mean():.3f}")

    # LightGBM hyper-parameter search: time-series CV on the training portion only
    grid = list(ParameterGrid({"n_estimators": [200, 400, 800], "learning_rate": [0.05],
                               "num_leaves": [7, 15, 31], "min_child_samples": [100, 300]}))
    tscv = TimeSeriesSplit(n_splits=3)
    best_cv, best_params = -1.0, None
    for params in grid:
        fold_f1 = []
        for fit_idx, val_idx in tscv.split(Itr):
            m = make_lgbm(params, SEED).fit(Itr.iloc[fit_idx], ytr[fit_idx])
            fold_f1.append(f1_score(ytr[val_idx], m.predict(Itr.iloc[val_idx]), average="macro"))
        cv = float(np.mean(fold_f1))
        print(f"  LGBM CV {params}: macro-F1={cv:.4f}")
        if cv > best_cv:
            best_cv, best_params = cv, params
    print(f"best LightGBM params {best_params} (CV macro-F1={best_cv:.4f}) [{time.time()-t0:.0f}s]")

    # Test set used once: fit each model on full training portion, evaluate per seed
    def run(make, seed, Fa, Fb):
        m = make(seed).fit(Fa, ytr)
        pred, proba = m.predict(Fb), m.predict_proba(Fb)[:, 1]
        return {"macro_f1": f1_score(yte, pred, average="macro"),
                "accuracy": accuracy_score(yte, pred), "logloss": log_loss(yte, proba)}

    base = [run(make_mlp, s, Ftr, Fte) for s in SEEDS]
    imp = [run(lambda s: make_lgbm(best_params, s), s, Itr, Ite) for s in SEEDS]
    for name, runs in [("baseline MLP", base), ("improved LGBM", imp)]:
        print(f"{name}: " + ", ".join(f"{k}={np.mean([r[k] for r in runs]):.4f}"
                                      for k in ["macro_f1", "accuracy", "logloss"]))

    def agg(runs, k):
        v = np.array([r[k] for r in runs])
        return float(v.mean()), float(v.std())

    b_f1, b_sd = agg(base, "macro_f1")
    i_f1, i_sd = agg(imp, "macro_f1")
    if not np.isfinite([b_f1, i_f1]).all() or len(yte) < 100 or len(np.unique(yte)) < 2:
        print("ERROR: no meaningful metrics produced")
        sys.exit(1)

    # Optional chart
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP baseline", "LightGBM improved"], [b_f1, i_f1], yerr=[b_sd, i_sd],
           capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("Test macro-F1 (mean ± sd over 3 seeds)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)

    results = {
        "question": "Can a class-balanced, CV-tuned LightGBM model with an interpretable feature set beat a standard MLP at classifying high-congestion intersection states from traffic-context features on the public UCI Metro Interstate Traffic Volume data?",
        "method": "Chronological 80/20 split; target = traffic_volume above training median; MLP baseline vs class-balanced LightGBM with engineered time-of-week/rush-hour/weather features, tuned by TimeSeriesSplit CV on training data; test set used once; seeds 42-44.",
        "dataset": {
            "name": "UCI Metro Interstate Traffic Volume (id=492)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(df)), "n_train": int(len(Ftr)), "n_test": int(len(Fte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test set)",
        "higher_is_better": True,
        "baseline": "Standard scikit-learn MLPClassifier (64-32 hidden units, early stopping) on standardised numeric and one-hot categorical features.",
        "improved": "Class-balanced LightGBM (class_weight='balanced', subsampling) on base plus engineered features (hour-of-week, cyclical hour, rush-hour, night, clear-sky, precipitation flags), with hyper-parameters chosen by 3-fold time-series CV on the training portion.",
        "metrics": {
            "baseline_score": b_f1,
            "improved_score": i_f1,
            "improvement": i_f1 - b_f1,
            "baseline_score_std": b_sd,
            "improved_score_std": i_sd,
            "baseline_accuracy": agg(base, "accuracy")[0],
            "improved_accuracy": agg(imp, "accuracy")[0],
            "baseline_logloss": agg(base, "logloss")[0],
            "improved_logloss": agg(imp, "logloss")[0],
            "lgbm_best_cv_macro_f1": best_cv,
            "n_seeds": len(SEEDS),
        },
        "notes": (f"Selected LightGBM params {best_params} by time-series CV on training data only. "
                  "Improved model uses extra engineered features that the baseline does not receive; "
                  "baseline features and MLP are unchanged. "
                  "Label is a tabular stand-in for the paper's camera-based signal state, not its RGB footage. "
                  "Chronological split means the test period is later than training (distribution shift). "
                  "MLP is untuned by design (fixed architecture), so tuning budget is not equal. "
                  "Seeds vary model randomness only, not the split. Class-balanced LightGBM probabilities "
                  "are not calibrated to the natural class prior, which can affect log-loss. "
                  f"Total runtime {time.time()-t0:.0f}s."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"improvement (macro-F1) = {i_f1 - b_f1:+.4f}; results written to results.json "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
