"""
RouteMind - extra analysis for report evidence. TRAINING DATA ONLY.

The test set is never scored and no frozen model, config value or
threshold is changed. Nothing here is saved as a model.

  1. Rule baseline: "late if Shipping Mode is First Class or Second Class"
  2. Leakage demo: the same Random Forest with two leaked columns
     deliberately added back (throwaway model, to show why they were removed)
  3. MLP sweep: learning rate x batch size (justifies the config values)
  4. class_weight check: None vs "balanced" (explains the calibration result)

Same 5 folds as train.py (StratifiedKFold, shuffle, random_state 42).
Writes logs/extra_analysis_<timestamp>.log and outputs/extra_*.csv
"""

import copy
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from features import build_features, load_config, make_preprocessor, split_data

LEAKED = {"numeric": "Days for shipping (real)", "categorical": "Delivery Status"}
RULE_MODES = ["First Class", "Second Class"]
MLP_LEARNING_RATES = [0.01, 0.001, 0.0001]
MLP_BATCH_SIZES = [64, 256]
METRICS = ["recall", "precision", "accuracy"]


def three_scores(y, pred):
    return {"recall": recall_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "accuracy": accuracy_score(y, pred),
            "flagged": float(np.mean(pred))}


def main():
    cfg = load_config()
    d, paths = cfg["data"], cfg["paths"]
    out = paths["output_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(paths["log_dir"], f"extra_analysis_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger()

    # Read the raw file here (not load_dataset) because part 2 needs the two
    # leaked columns. The same cancelled-order filter is applied.
    raw = pd.read_csv(d["path"], encoding=d["encoding"])
    rule = d["exclude_rows"]
    raw = raw[~raw[rule["column"]].isin(rule["values"])]
    y = raw[d["label"]]
    X = build_features(raw, cfg)                 # the normal 15 features
    X_leak = X.copy()                            # + 2 leaked columns (part 2 only)
    X_leak[LEAKED["numeric"]] = raw[LEAKED["numeric"]]
    X_leak[LEAKED["categorical"]] = raw[LEAKED["categorical"]]

    # Same split as every other script; the test part is discarded unseen.
    X_train, _, y_train, _ = split_data(X, y, cfg)
    X_leak_train = X_leak.loc[X_train.index]
    y_arr = y_train.to_numpy()
    seed = cfg["split"]["random_state"]
    threshold = cfg["evaluation"]["decision_threshold"]

    def folds():
        return StratifiedKFold(n_splits=cfg["evaluation"]["cv_folds"],
                               shuffle=True, random_state=seed)

    def rf_pipeline(settings, config=cfg):
        return Pipeline([("prep", make_preprocessor(config, scale_numbers=False)),
                         ("model", RandomForestClassifier(**settings))])

    log.info("=" * 66)
    log.info("ROUTEMIND EXTRA ANALYSIS (training data only; test set not scored)")
    log.info("=" * 66)
    log.info(f"Training orders: {len(X_train):,} | late rate {y_arr.mean():.1%} "
             f"| 5 folds, random_state {seed}")

    # Out-of-fold risk for the frozen RF settings (used in parts 1 and 4)
    rf_settings = dict(cfg["random_forest"])
    risk_bal = cross_val_predict(rf_pipeline(rf_settings), X_train, y_train,
                                 cv=folds(), method="predict_proba")[:, 1]
    rf_cv = three_scores(y_arr, (risk_bal >= threshold).astype(int))

    # ---- 1. Rule baseline ------------------------------------------------
    rule_pred = X_train["Shipping Mode"].isin(RULE_MODES).astype(int).to_numpy()
    rule_scores = three_scores(y_arr, rule_pred)
    t1 = pd.DataFrame({
        f"Rule: late if Shipping Mode in {RULE_MODES}": rule_scores,
        "Random Forest (5-fold CV, threshold 0.5)": rf_cv,
        "Flag every order": three_scores(y_arr, np.ones(len(y_arr), dtype=int)),
    }).T.round(4)
    agree = float(np.mean(rule_pred == (risk_bal >= threshold)))
    log.info("\n1. RULE BASELINE vs RANDOM FOREST (training set)\n" + t1.to_string())
    log.info(f"   The rule and the Random Forest give the same decision for "
             f"{agree:.2%} of training orders.")
    t1.to_csv(os.path.join(out, "extra_rule_baseline.csv"))

    # ---- 2. Leakage demo (throwaway model, never saved) --------------------
    leak_cfg = copy.deepcopy(cfg)
    leak_cfg["data"]["numeric_features"].append(LEAKED["numeric"])
    leak_cfg["data"]["categorical_features"].append(LEAKED["categorical"])
    rows = {}
    for label, data, config in [
            ("Without leaked columns (our model)", X_train, cfg),
            ("WITH leaked columns added back (demo only)", X_leak_train, leak_cfg)]:
        cv = cross_validate(rf_pipeline(rf_settings, config), data, y_train,
                            cv=folds(), scoring=METRICS)
        rows[label] = {m: cv[f"test_{m}"].mean() for m in METRICS}
    t2 = pd.DataFrame(rows).T.round(4)
    log.info("\n2. LEAKAGE DEMO (5-fold CV, same RF settings)\n" + t2.to_string())
    log.info(f"   Leaked columns added for this demo: {list(LEAKED.values())}. "
             f"The model is not saved;\n   the leakage guard in features.py is unchanged.")
    t2.to_csv(os.path.join(out, "extra_leakage_demo.csv"))

    # ---- 3. MLP sweep --------------------------------------------------------
    log.info("\n3. MLP SWEEP (5-fold CV; other settings as in config.yaml)")
    base = dict(cfg["mlp"])
    base["hidden_layer_sizes"] = tuple(base["hidden_layer_sizes"])
    rows = []
    for lr in MLP_LEARNING_RATES:
        for batch in MLP_BATCH_SIZES:
            settings = {**base, "learning_rate_init": lr, "batch_size": batch}
            mlp = Pipeline([("prep", make_preprocessor(cfg, scale_numbers=True)),
                            ("model", MLPClassifier(**settings))])
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                cv = cross_validate(mlp, X_train, y_train, cv=folds(),
                                    scoring=METRICS, return_estimator=True,
                                    n_jobs=-1)
            epochs = [est.named_steps["model"].n_iter_ for est in cv["estimator"]]
            row = {"learning_rate": lr, "batch_size": batch,
                   "val_accuracy": cv["test_accuracy"].mean(),
                   "val_recall": cv["test_recall"].mean(),
                   "val_precision": cv["test_precision"].mean(),
                   "epochs_mean": float(np.mean(epochs)),
                   "epochs_per_fold": str(epochs),
                   "hit_max_epochs": sum(e >= base["max_iter"] for e in epochs),
                   "seconds": round(time.time() - t)}
            rows.append(row)
            chosen = " <- config.yaml" if (lr == cfg["mlp"]["learning_rate_init"]
                                           and batch == cfg["mlp"]["batch_size"]) else ""
            log.info(f"   lr {lr:<7} batch {batch:<4} | accuracy {row['val_accuracy']:.4f} "
                     f"recall {row['val_recall']:.4f} precision {row['val_precision']:.4f} "
                     f"| epochs {epochs} | {row['seconds']}s{chosen}")
    t3 = pd.DataFrame(rows)
    t3.round(4).to_csv(os.path.join(out, "extra_mlp_sweep.csv"), index=False)
    log.info(f"   max epochs allowed: {base['max_iter']}; early stopping patience "
             f"{base['n_iter_no_change']}")

    # ---- 4. class_weight check ---------------------------------------------
    risk_none = cross_val_predict(
        rf_pipeline({**rf_settings, "class_weight": None}), X_train, y_train,
        cv=folds(), method="predict_proba")[:, 1]
    rows = {}
    for label, risk in [("class_weight = balanced (frozen model)", risk_bal),
                        ("class_weight = None", risk_none)]:
        s = three_scores(y_arr, (risk >= threshold).astype(int))
        s["mean_predicted_risk"] = float(risk.mean())
        s["brier_score"] = brier_score_loss(y_arr, risk)
        rows[label] = s
    t4 = pd.DataFrame(rows).T.round(4)
    log.info("\n4. CLASS_WEIGHT CHECK (5-fold out-of-fold, threshold 0.5)\n" + t4.to_string())
    log.info(f"   Actual late rate in training: {y_arr.mean():.4f} "
             f"(a calibrated model's mean predicted risk should be close to this)")
    t4.to_csv(os.path.join(out, "extra_class_weight.csv"))

    log.info(f"\nCSVs -> {out}/extra_rule_baseline.csv, extra_leakage_demo.csv, "
             f"extra_mlp_sweep.csv, extra_class_weight.csv")
    log.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
