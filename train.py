"""
RouteMind - Step 1.3: Train the late-delivery risk models.

  1. Load the data (leakage columns and cancelled orders removed)
  2. Split 80/20, keeping the late/on-time ratio the same in both parts
  3. Random Forest (main model): 5-fold cross-validation, then fit
  4. MLP (comparison model): fit and record the loss for every epoch
  5. Save both models and the untouched test set for evaluate.py

Every run writes logs/train_<timestamp>.log
"""

import logging
import os
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from features import (build_features, forbidden_columns, load_config,
                      load_dataset, make_preprocessor, split_data)


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"train_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(path, encoding="utf-8"),
                  logging.StreamHandler()],
    )
    return path


def main():
    cfg = load_config()
    paths = cfg["paths"]
    log_path = setup_logging(paths["log_dir"])
    os.makedirs("models", exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)

    log = logging.getLogger()
    log.info("=" * 60)
    log.info("ROUTEMIND TRAINING RUN")
    log.info("=" * 60)

    # 1. Data
    df = load_dataset(cfg)
    X = build_features(df, cfg)
    y = df[cfg["data"]["label"]]
    log.info(f"Orders after cleaning: {len(X):,}")
    log.info(f"Features used ({X.shape[1]}): {list(X.columns)}")
    # Verification (amendment D): leakage / filter-only columns are not inputs.
    for col in forbidden_columns(cfg):
        log.info(f"  CHECK '{col}' in features? {col in X.columns}")
    log.info(f"Late rate: {y.mean():.1%}")

    # 2. Split
    s = cfg["split"]
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    log.info(f"Preprocessing settings: {cfg['preprocessing']}")
    log.info(f"Train: {len(X_train):,} orders | Test: {len(X_test):,} orders")

    # 3. Random Forest
    rf_cfg = cfg["random_forest"]
    log.info("-" * 60)
    log.info(f"RANDOM FOREST settings: {rf_cfg}")
    rf = Pipeline([
        ("prep", make_preprocessor(cfg, scale_numbers=False)),
        ("model", RandomForestClassifier(**rf_cfg)),
    ])

    folds = cfg["evaluation"]["cv_folds"]
    log.info(f"Running {folds}-fold cross-validation on the training set...")
    t = time.time()
    cv = cross_validate(
        rf, X_train, y_train,
        cv=StratifiedKFold(n_splits=folds, shuffle=True,
                           random_state=s["random_state"]),
        scoring=["recall", "precision", "accuracy"],
        return_train_score=True, n_jobs=1)
    # Verification (amendment C): training and validation scores come from
    # separate keys. train_* = scored on the 4 folds the model learned from;
    # test_* = scored on the 1 held-out fold it never saw.
    log.info(f"  CHECK score keys returned: {sorted(k for k in cv if '_' in k and k.split('_')[0] in ('train', 'test'))}")
    for m in ["recall", "precision", "accuracy"]:
        tr, va = cv[f"train_{m}"], cv[f"test_{m}"]
        log.info(f"  {m:<9} train {tr.mean():.3f} | validation "
                 f"{va.mean():.3f} (+/- {va.std():.3f}) | gap {tr.mean() - va.mean():.3f}")
    log.info(f"  Cross-validation took {time.time() - t:.0f}s")

    t = time.time()
    rf.fit(X_train, y_train)
    log.info(f"Random Forest fitted on full training set in {time.time() - t:.0f}s")
    joblib.dump(rf, paths["rf_model_out"], compress=3)
    log.info(f"Saved -> {paths['rf_model_out']}")

    # 4. MLP comparison
    mlp_cfg = dict(cfg["mlp"])
    mlp_cfg["hidden_layer_sizes"] = tuple(mlp_cfg["hidden_layer_sizes"])
    log.info("-" * 60)
    log.info(f"MLP settings: {mlp_cfg}")
    mlp = Pipeline([
        ("prep", make_preprocessor(cfg, scale_numbers=True)),
        ("model", MLPClassifier(**mlp_cfg)),
    ])
    t = time.time()
    mlp.fit(X_train, y_train)
    net = mlp.named_steps["model"]
    log.info(f"MLP trained for {net.n_iter_} epochs in {time.time() - t:.0f}s")
    for epoch, (loss, val) in enumerate(
            zip(net.loss_curve_, net.validation_scores_), start=1):
        log.info(f"  epoch {epoch:>2}: training loss {loss:.4f} | "
                 f"validation accuracy {val:.4f}")
    if net.n_iter_ < mlp_cfg["max_iter"]:
        log.info(f"  Early stopping: no improvement for "
                 f"{mlp_cfg['n_iter_no_change']} epochs")
    joblib.dump(mlp, paths["mlp_model_out"], compress=3)
    log.info(f"Saved -> {paths['mlp_model_out']}")

    # Loss curve chart (evidence for the epochs discussion)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, net.n_iter_ + 1)
    ax1.plot(epochs, net.loss_curve_, marker="o", label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, net.validation_scores_, marker="s", color="tab:orange",
             label="Validation accuracy")
    ax2.set_ylabel("Validation accuracy")
    ax1.set_title("MLP training: loss and validation accuracy per epoch")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.85))
    fig.tight_layout()
    chart = os.path.join(paths["output_dir"], "mlp_training_curve.png")
    fig.savefig(chart, dpi=150)
    log.info(f"Saved chart -> {chart}")

    # 5. Keep the test set aside for evaluate.py
    joblib.dump((X_test, y_test), paths["test_set_out"], compress=3)
    log.info(f"Saved untouched test set -> {paths['test_set_out']}")
    log.info("=" * 60)
    log.info(f"Training complete. Log: {log_path}")


if __name__ == "__main__":
    main()
