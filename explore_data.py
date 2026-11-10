"""
RouteMind - Step 1.2: Explore the DataCo dataset.

Answers four questions before we train anything:
  1. How big is the dataset and what columns does it have?
  2. How balanced is the label (late vs on time)?
  3. Are there missing values in the columns we plan to use?
  4. Which columns leak the answer (only known after delivery)?

Results are printed and saved to logs/explore_<timestamp>.log
"""

import os
from datetime import datetime

import pandas as pd
import yaml


def log(msg, lines):
    print(msg)
    lines.append(msg)


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    label = data_cfg["label"]
    lines = []

    df = pd.read_csv(data_cfg["path"], encoding=data_cfg["encoding"])

    # 1. Size and columns
    log("=" * 60, lines)
    log("1. DATASET SIZE", lines)
    log("=" * 60, lines)
    log(f"Rows: {len(df):,}   Columns: {df.shape[1]}", lines)
    log("\nAll columns:", lines)
    for col in df.columns:
        log(f"  - {col}  ({df[col].dtype})", lines)

    # 2. Label balance
    log("\n" + "=" * 60, lines)
    log(f"2. LABEL BALANCE ({label})", lines)
    log("=" * 60, lines)
    counts = df[label].value_counts().sort_index()
    for value, n in counts.items():
        name = "late" if value == 1 else "on time"
        log(f"  {value} ({name}): {n:,}  ({n / len(df):.1%})", lines)

    # 3. Missing values in planned features
    log("\n" + "=" * 60, lines)
    log("3. MISSING VALUES IN PLANNED FEATURES", lines)
    log("=" * 60, lines)
    planned = (data_cfg["numeric_features"] + data_cfg["categorical_features"]
               + [data_cfg["date_column"]])
    missing_cols = [c for c in planned if c not in df.columns]
    if missing_cols:
        log(f"  Not found in dataset: {missing_cols}", lines)
    for col in [c for c in planned if c in df.columns]:
        log(f"  {col}: {df[col].isna().sum()} missing", lines)

    # 4. Leakage check: how well does each suspect column alone reveal the label?
    log("\n" + "=" * 60, lines)
    log("4. LEAKAGE CHECK", lines)
    log("=" * 60, lines)

    real = df["Days for shipping (real)"]
    sched = df["Days for shipment (scheduled)"]
    rule = (real > sched).astype(int)
    log("  Rule 'real days > scheduled days' matches the label for "
        f"{(rule == df[label]).mean():.1%} of orders", lines)
    log("  -> the label is calculated from real shipping days, "
        "so that column must be removed.", lines)

    log("\n  Delivery Status vs label:", lines)
    log(pd.crosstab(df["Delivery Status"], df[label]).to_string(), lines)

    log("\n  Order Status vs label:", lines)
    log(pd.crosstab(df["Order Status"], df[label]).to_string(), lines)

    # 5. Shipping Mode quirk: is the high late rate explained by short
    #    promised (scheduled) windows? Measured from our own data, not assumed.
    log("\n" + "=" * 60, lines)
    log("5. SHIPPING MODE: LATE RATE vs SCHEDULED AND REAL DAYS", lines)
    log("=" * 60, lines)
    shipped = df[df["Delivery Status"] != "Shipping canceled"]
    by_mode = shipped.groupby("Shipping Mode").agg(
        orders=(label, "size"),
        late_rate=(label, "mean"),
        scheduled_days_avg=("Days for shipment (scheduled)", "mean"),
        scheduled_days_values=("Days for shipment (scheduled)",
                               lambda s: sorted(s.unique().tolist())),
        real_days_avg=("Days for shipping (real)", "mean"),
    )
    by_mode["late_rate"] = by_mode["late_rate"].map("{:.1%}".format)
    by_mode["scheduled_days_avg"] = by_mode["scheduled_days_avg"].round(2)
    by_mode["real_days_avg"] = by_mode["real_days_avg"].round(2)
    log("  (cancelled orders excluded, same as training)", lines)
    log(by_mode.to_string(), lines)
    log("  Note: real days are shown only to explain the quirk; they are a\n"
        "  leakage column and are never used as a model feature.", lines)

    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg["paths"]["log_dir"], f"explore_{stamp}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved log to {path}")


if __name__ == "__main__":
    main()
