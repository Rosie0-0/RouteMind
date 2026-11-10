"""
RouteMind - Step 3: max_depth validation curve for the Random Forest.

Question answered: how deep should the trees be allowed to grow?
  - too shallow -> the model is too simple (underfitting)
  - too deep    -> the model memorises the training data (overfitting):
                   training score rises while validation score does not

Each max_depth value is tested with 5-fold cross-validation on the
TRAINING set only. The test set is never loaded here.

Writes logs/tune_<timestamp>.log, outputs/max_depth_curve.png and .csv
"""

import logging
import os
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import (build_features, load_config, load_dataset,
                      make_preprocessor, split_data)

METRICS = ["recall", "precision", "accuracy"]


def main():
    cfg = load_config()
    paths = cfg["paths"]
    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(paths["log_dir"], f"tune_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger()

    # Training data only: the test part of the split is thrown away here.
    df = load_dataset(cfg)
    X = build_features(df, cfg)
    y = df[cfg["data"]["label"]]
    X_train, _, y_train, _ = split_data(X, y, cfg)

    folds = cfg["evaluation"]["cv_folds"]
    seed = cfg["split"]["random_state"]
    depths = cfg["tuning"]["max_depth_values"]
    base = dict(cfg["random_forest"])

    log.info("=" * 60)
    log.info("ROUTEMIND max_depth VALIDATION CURVE")
    log.info("=" * 60)
    log.info(f"Training orders: {len(X_train):,} (test set not loaded)")
    log.info(f"{folds}-fold cross-validation, depths tested: {depths}")
    log.info(f"Other RF settings held fixed: "
             f"{ {k: v for k, v in base.items() if k != 'max_depth'} }")

    rows = []
    for depth in depths:
        settings = {**base, "max_depth": depth}
        model = Pipeline([
            ("prep", make_preprocessor(cfg, scale_numbers=False)),
            ("model", RandomForestClassifier(**settings)),
        ])
        t = time.time()
        cv = cross_validate(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            scoring=METRICS, return_train_score=True, n_jobs=1)
        row = {"max_depth": "None" if depth is None else str(depth),
               "seconds": round(time.time() - t)}
        for m in METRICS:
            row[f"train_{m}"] = cv[f"train_{m}"].mean()
            row[f"val_{m}"] = cv[f"test_{m}"].mean()
            row[f"val_{m}_std"] = cv[f"test_{m}"].std()
            row[f"gap_{m}"] = row[f"train_{m}"] - row[f"val_{m}"]
        rows.append(row)
        log.info(f"max_depth={row['max_depth']:<4} | "
                 f"recall train {row['train_recall']:.3f} val {row['val_recall']:.3f} | "
                 f"precision train {row['train_precision']:.3f} val {row['val_precision']:.3f} | "
                 f"accuracy train {row['train_accuracy']:.3f} val {row['val_accuracy']:.3f} | "
                 f"{row['seconds']}s")

    table = pd.DataFrame(rows).set_index("max_depth")
    csv_path = os.path.join(paths["output_dir"], "max_depth_curve.csv")
    table.round(4).to_csv(csv_path)

    log.info("-" * 60)
    log.info("Train minus validation gap (bigger gap = more overfitting):")
    log.info("\n" + table[[f"gap_{m}" for m in METRICS]].round(4).to_string())

    # Chart: one panel per metric, training vs validation
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, m in zip(axes, METRICS):
        ax.plot(table.index, table[f"train_{m}"], marker="o", label="Training")
        ax.plot(table.index, table[f"val_{m}"], marker="s", label="Validation")
        ax.fill_between(table.index,
                        table[f"val_{m}"] - table[f"val_{m}_std"],
                        table[f"val_{m}"] + table[f"val_{m}_std"], alpha=0.2)
        ax.set_title(m.capitalize())
        ax.set_xlabel("max_depth")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score (5-fold CV mean)")
    axes[0].legend()
    fig.suptitle("Random Forest validation curve: effect of max_depth "
                 "(training data only)")
    fig.tight_layout()
    chart = os.path.join(paths["output_dir"], "max_depth_curve.png")
    fig.savefig(chart, dpi=150)

    log.info(f"Saved -> {chart}, {csv_path}")
    log.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
