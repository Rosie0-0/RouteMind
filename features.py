"""
RouteMind - shared data preparation.

Used by train.py (whole dataset) and predict.py (one new order), so both
prepare data in exactly the same way.
"""

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATE_FEATURES = ["order_month", "order_weekday", "order_hour"]


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(cfg):
    """Read the CSV and remove rows whose label is unreliable."""
    d = cfg["data"]
    df = pd.read_csv(d["path"], encoding=d["encoding"])

    rule = d.get("exclude_rows")
    if rule:
        df = df[~df[rule["column"]].isin(rule["values"])]

    # Leakage columns are dropped here so they can never reach the model.
    return df.drop(columns=d["leakage_columns"])


def build_features(df, cfg):
    """Turn raw order columns into the model's input table X."""
    d = cfg["data"]
    X = df[d["numeric_features"] + d["categorical_features"]].copy()

    # An order date is not useful as raw text; month, weekday and hour are.
    # errors="coerce": a missing or badly written date becomes NaT (empty)
    # instead of crashing; the imputer then fills month/weekday/hour.
    dates = pd.to_datetime(df[d["date_column"]], format=d["date_format"],
                           errors="coerce")
    X["order_month"] = dates.dt.month
    X["order_weekday"] = dates.dt.weekday
    X["order_hour"] = dates.dt.hour

    # Safety guard: stop immediately if a leakage or filter-only column
    # ever ends up in the model input (e.g. someone edits config.yaml).
    forbidden = set(d["leakage_columns"]) | {d["label"]}
    rule = d.get("exclude_rows")
    if rule:
        forbidden.add(rule["column"])
    leaked = forbidden & set(X.columns)
    if leaked:
        raise ValueError(f"Leakage guard: forbidden columns in features: {leaked}")
    return X


def forbidden_columns(cfg):
    """Columns that must never be model inputs (for logging/verification)."""
    d = cfg["data"]
    cols = list(d["leakage_columns"]) + [d["label"]]
    rule = d.get("exclude_rows")
    if rule and rule["column"] not in cols:
        cols.append(rule["column"])
    return cols


def numeric_columns(cfg):
    return cfg["data"]["numeric_features"] + DATE_FEATURES


def categorical_columns(cfg):
    return cfg["data"]["categorical_features"]


def split_data(X, y, cfg):
    """The one train/test split every script uses. Same random_state ->
    the same 34,553 test orders every time, so the test set stays untouched."""
    s = cfg["split"]
    return train_test_split(X, y, test_size=s["test_size"],
                            random_state=s["random_state"],
                            stratify=y if s["stratify"] else None)


def make_preprocessor(cfg, scale_numbers):
    """Raw order columns -> numbers the model can use.

    Numbers:    fill gaps with the median; scale only for the MLP (trees
                split on thresholds so scale does not matter to them, but a
                neural network learns badly when one input is 1000x bigger).
    Categories: fill gaps with "Unknown", then one-hot encode. A category
                never seen in training (e.g. a new region) becomes all zeros
                instead of crashing.

    It sits inside the model Pipeline, so during cross-validation the
    median and scaling are learned from the training folds only.
    """
    p = cfg["preprocessing"]
    num_steps = [("impute", SimpleImputer(strategy=p["numeric_impute"]))]
    if scale_numbers:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant",
                                 fill_value=p["categorical_impute"])),
        ("onehot", OneHotEncoder(handle_unknown=p["unknown_categories"])),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric_columns(cfg)),
        ("cat", Pipeline(cat_steps), categorical_columns(cfg)),
    ])
