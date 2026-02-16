import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"id", "study_interval", "day_in_study"}.issubset(df.columns):
        return df

    out = df.sort_values(["id", "study_interval", "day_in_study"]).copy()
    out["cycle_sin_28"] = np.sin(2 * np.pi * out["day_in_study"] / 28.0)
    out["cycle_cos_28"] = np.cos(2 * np.pi * out["day_in_study"] / 28.0)

    lag_candidates = safe_columns(
        out,
        [
            "lh",
            "estrogen",
            "pdg",
            "wrist_temp_mean",
            "nightly_temp_mean",
            "rmssd_mean",
            "stress_score_mean",
            "stress_score_max",
            "sleep_points_mean",
            "responsiveness_points_mean",
            "exertion_points_mean",
        ],
    )
    for col in lag_candidates:
        out[f"{col}_lag1"] = out.groupby(["id", "study_interval"])[col].shift(1)
        out[f"{col}_delta"] = out[col] - out[f"{col}_lag1"]
        shifted = out.groupby(["id", "study_interval"])[col].shift(1)
        out[f"{col}_roll3_mean"] = (
            shifted.groupby([out["id"], out["study_interval"]])
            .rolling(window=3, min_periods=2)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        out[f"{col}_roll3_std"] = (
            shifted.groupby([out["id"], out["study_interval"]])
            .rolling(window=3, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

    for col in safe_columns(out, ["flow_volume", "flow_color"]):
        out[f"{col}_lag1"] = out.groupby(["id", "study_interval"])[col].shift(1)
    if "phase" in out.columns:
        out["phase_lag1"] = out.groupby(["id", "study_interval"])["phase"].shift(1)

    return out


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_mode(
    mode_name: str,
    df_mode: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    label_col = "phase"
    y = df_mode[label_col].astype(str)

    drop_cols = [label_col]
    for c in ["id", "study_interval"]:
        if c in df_mode.columns:
            drop_cols.append(c)
    X = df_mode.drop(columns=drop_cols, errors="ignore")

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            C=0.5,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    cv = GroupKFold(n_splits=5)

    rows = []
    for model_name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_result = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=g_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        rows.append(
            {
                "mode": mode_name,
                "model_name": model_name,
                "feature_count": len(X.columns),
                "cv_accuracy": float(np.mean(cv_result["test_accuracy"])),
                "cv_balanced_accuracy": float(np.mean(cv_result["test_balanced_accuracy"])),
                "cv_f1_macro": float(np.mean(cv_result["test_f1_macro"])),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(data_dir, "merged_women_data.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    if "phase" not in df.columns:
        raise ValueError("Column 'phase' is required.")
    df = df.dropna(subset=["phase"]).copy()

    # Build both modes using same row universe
    df_with_history = add_temporal_features(df)
    df_current_only = df.copy()

    # Use same grouped split for fair comparison
    groups = df["id"] if "id" in df.columns else pd.Series(np.arange(len(df)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["phase"], groups=groups))

    comp_a = evaluate_mode(
        mode_name="current_only_no_history",
        df_mode=df_current_only,
        train_idx=train_idx,
        test_idx=test_idx,
        groups=groups,
    )
    comp_b = evaluate_mode(
        mode_name="with_history_features",
        df_mode=df_with_history,
        train_idx=train_idx,
        test_idx=test_idx,
        groups=groups,
    )

    result = pd.concat([comp_a, comp_b], axis=0, ignore_index=True)
    result = result.sort_values(["mode", "test_f1_macro"], ascending=[True, False])

    out_csv = os.path.join(data_dir, "phase_history_vs_current_comparison.csv")
    result.to_csv(out_csv, index=False)

    best_per_mode = (
        result.sort_values("test_f1_macro", ascending=False)
        .groupby("mode", as_index=False)
        .first()
        .to_dict(orient="records")
    )

    report = {
        "row_count": int(df.shape[0]),
        "phase_distribution": df["phase"].value_counts().to_dict(),
        "comparison_file": out_csv,
        "best_per_mode": best_per_mode,
        "notes": [
            "Both modes use the exact same grouped train/test split.",
            "This isolates impact of history features on phase prediction quality.",
        ],
    }
    out_json = os.path.join(data_dir, "phase_history_vs_current_report.json")
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
