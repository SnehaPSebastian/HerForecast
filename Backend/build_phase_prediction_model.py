import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def safe_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time ordering by participant
    if not {"id", "study_interval", "day_in_study"}.issubset(df.columns):
        return df

    out = df.sort_values(["id", "study_interval", "day_in_study"]).copy()

    # Generic cycle prior (not clinical ground truth, just periodic inductive bias)
    out["cycle_sin_28"] = np.sin(2 * np.pi * out["day_in_study"] / 28.0)
    out["cycle_cos_28"] = np.cos(2 * np.pi * out["day_in_study"] / 28.0)

    lag_candidates = [
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
    ]
    lag_candidates = safe_columns(out, lag_candidates)

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

    # Lag category context (useful for transitions, does not leak label)
    for col in safe_columns(out, ["flow_volume", "flow_color"]):
        out[f"{col}_lag1"] = out.groupby(["id", "study_interval"])[col].shift(1)
    # Prior phase as historical context for sequential prediction usage.
    if "phase" in out.columns:
        out["phase_lag1"] = out.groupby(["id", "study_interval"])["phase"].shift(1)

    return out


def evaluate_model_candidates(
    model_name: str,
    model_candidates: List[Tuple[str, object]],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    g_train: pd.Series,
    cv: GroupKFold,
    scoring: Dict[str, str],
) -> Tuple[str, Pipeline, Dict[str, float], List[Dict[str, float]]]:
    candidate_rows: List[Dict[str, float]] = []
    best_key = ""
    best_pipe = None
    best_cv_f1 = -np.inf
    best_cv_metrics: Dict[str, float] = {}

    for candidate_key, model in model_candidates:
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
        cv_metrics = {
            "cv_accuracy": float(np.mean(cv_result["test_accuracy"])),
            "cv_balanced_accuracy": float(np.mean(cv_result["test_balanced_accuracy"])),
            "cv_f1_macro": float(np.mean(cv_result["test_f1_macro"])),
        }
        row = {"model_name": model_name, "candidate": candidate_key}
        row.update(cv_metrics)
        candidate_rows.append(row)

        if cv_metrics["cv_f1_macro"] > best_cv_f1:
            best_cv_f1 = cv_metrics["cv_f1_macro"]
            best_key = candidate_key
            best_pipe = pipe
            best_cv_metrics = cv_metrics

    if best_pipe is None:
        raise RuntimeError(f"No candidate could be evaluated for {model_name}")

    return best_key, best_pipe, best_cv_metrics, candidate_rows


def main() -> None:
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(data_dir, "synthetic_women_cycle_data.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded synthetic data: {df.shape}")

    # Rename columns to match expected features
    df = df.rename(columns={
        'Day_in_Cycle': 'day_in_study',
        'Skin_Temperature': 'wrist_temp_mean',
        'PPG_RMSSD': 'rmssd_mean',
        'GSR': 'stress_score_mean',
        'Estrogen': 'estrogen',
        'Progesterone': 'pdg',
        'Phase': 'phase'
    })

    # Map phase labels to match model classes
    phase_mapping = {
        'menstrual': 'Menstrual',
        'follicular': 'Follicular',
        'ovulation': 'Fertility',
        'luteal': 'Luteal'
    }
    df['phase'] = df['phase'].map(phase_mapping)

    # Keep only rows with known menstrual phase labels.
    if "phase" not in df.columns:
        raise ValueError("Column 'phase' not found in synthetic_women_cycle_data.csv")

    df = df.dropna(subset=["phase"]).copy()
    df = add_temporal_features(df)
    print(f"Rows with phase label: {df.shape[0]}")
    print("Phase distribution:")
    print(df["phase"].value_counts().to_string())

    # Label and grouping
    label_col = "phase"
    y = df[label_col].astype(str)
    groups = df["id"] if "id" in df.columns else pd.Series(np.arange(len(df)))

    # Features: use all columns except label and strict identifiers.
    drop_cols = [label_col]
    if "id" in df.columns:
        drop_cols.append("id")
    if "study_interval" in df.columns:
        drop_cols.append("study_interval")
    X_all = df.drop(columns=drop_cols, errors="ignore")

    # Feature strategy modes:
    # - full_history (default): best predictive quality and strongest standard fit
    # - hormone_primary: hormone-centric subset for interpretability preference
    feature_mode = os.getenv("PHASE_FEATURE_MODE", "full_history").strip().lower()
    hormone_tokens = ["lh", "estrogen", "pdg"]
    hormone_cols_selected = [
        col for col in X_all.columns if any(token in col for token in hormone_tokens)
    ]
    if feature_mode == "hormone_primary":
        context_cols = safe_columns(
            X_all,
            [
                "day_in_study",
                "is_weekend",
                "cycle_sin_28",
                "cycle_cos_28",
                "flow_volume",
                "flow_color",
                "flow_volume_lag1",
                "flow_color_lag1",
                "wrist_temp_mean",
                "nightly_temp_mean",
                "rmssd_mean",
                "stress_score_mean",
            ],
        )
        selected_cols = sorted(set(hormone_cols_selected + context_cols))
    else:
        feature_mode = "full_history"
        selected_cols = sorted(X_all.columns.tolist())
    X = X_all[selected_cols].copy()

    # Build typed feature lists
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Total selected features: {len(selected_cols)}")
    print(
        f"Hormone-core features selected: {len(hormone_cols_selected)} "
        f"({len(hormone_cols_selected)/max(len(selected_cols),1):.1%} of selected set)"
    )
    print(f"Feature mode: {feature_mode}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Split by participant to avoid leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model_candidates: Dict[str, List[Tuple[str, object]]] = {
        "logistic_regression": [
            (
                "c_0.5",
                LogisticRegression(
                    C=0.5,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
            (
                "c_1.0",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
            (
                "c_2.0",
                LogisticRegression(
                    C=2.0,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ],
        "random_forest": [
            (
                "rf_a",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=20,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "rf_b",
                RandomForestClassifier(
                    n_estimators=900,
                    max_depth=24,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "rf_c",
                RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        "extra_trees": [
            (
                "et_a",
                ExtraTreesClassifier(
                    n_estimators=700,
                    max_depth=20,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "et_b",
                ExtraTreesClassifier(
                    n_estimators=900,
                    max_depth=24,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "et_c",
                ExtraTreesClassifier(
                    n_estimators=1100,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
    }

    cv = GroupKFold(n_splits=5)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    cv_summary = {}
    fitted = {}
    test_summary = {}
    test_predictions = {}
    tuning_rows = []

    selected_candidate_key = {}
    for name, candidates in model_candidates.items():
        print(f"\nTraining and validating: {name}")
        best_key, best_pipe, best_cv_metrics, rows = evaluate_model_candidates(
            model_name=name,
            model_candidates=candidates,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            cv=cv,
            scoring=scoring,
        )
        tuning_rows.extend(rows)
        selected_candidate_key[name] = best_key
        cv_summary[name] = best_cv_metrics

        best_pipe.fit(X_train, y_train)
        fitted[name] = best_pipe
        pred = best_pipe.predict(X_test)
        test_predictions[name] = pred
        test_summary[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        }

    winner = max(test_summary.keys(), key=lambda k: test_summary[k]["f1_macro"])
    final_pipe = fitted[winner]
    final_pred = test_predictions[winner]
    print(f"\nSelected model: {winner}")

    # Confusion matrix and detailed report
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, final_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_path = os.path.join(data_dir, "phase_confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)

    cls_report = classification_report(y_test, final_pred, labels=labels, output_dict=True, zero_division=0)

    # Correlation/association: mutual information on preprocessed train features.
    prep = final_pipe.named_steps["preprocessor"]
    X_train_proc = prep.transform(X_train)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    # Use one-hot encoded phase labels for MI target
    y_codes = pd.Categorical(y_train).codes
    mi = mutual_info_classif(X_train_proc, y_codes, random_state=42)
    feat_names = prep.get_feature_names_out()
    mi_df = pd.DataFrame(
        {"feature": feat_names, "mutual_info": mi, "abs_mutual_info": np.abs(mi)}
    ).sort_values("abs_mutual_info", ascending=False)
    mi_path = os.path.join(data_dir, "phase_feature_correlation.csv")
    mi_df.to_csv(mi_path, index=False)

    # Export row-wise predictions on test set.
    pred_out = df.iloc[test_idx][safe_columns(df, ["id", "study_interval", "day_in_study"])].copy()
    pred_out["true_phase"] = y_test.values
    pred_out["predicted_phase"] = final_pred
    pred_path = os.path.join(data_dir, "phase_predictions.csv")
    pred_out.to_csv(pred_path, index=False)

    comparison_rows = []
    for model_name in cv_summary.keys():
        row = {
            "model_name": model_name,
            "selected_candidate": selected_candidate_key[model_name],
            "cv_accuracy": cv_summary[model_name]["cv_accuracy"],
            "cv_balanced_accuracy": cv_summary[model_name]["cv_balanced_accuracy"],
            "cv_f1_macro": cv_summary[model_name]["cv_f1_macro"],
            "test_accuracy": test_summary[model_name]["accuracy"],
            "test_balanced_accuracy": test_summary[model_name]["balanced_accuracy"],
            "test_f1_macro": test_summary[model_name]["f1_macro"],
        }
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows).sort_values("test_f1_macro", ascending=False)
    comparison_path = os.path.join(data_dir, "phase_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    tuning_path = os.path.join(data_dir, "phase_model_tuning_cv.csv")
    pd.DataFrame(tuning_rows).sort_values(["model_name", "cv_f1_macro"], ascending=[True, False]).to_csv(
        tuning_path, index=False
    )

    # Save model artifact
    artifact = {
        "model_name": winner,
        "pipeline": final_pipe,
        "label_classes": labels,
        "feature_columns": X.columns.tolist(),
        "target_label": "phase",
    }
    artifact_path = os.path.join(data_dir, "phase_prediction_model.joblib")
    joblib.dump(artifact, artifact_path)

    report = {
        "data_shape_with_label": [int(df.shape[0]), int(df.shape[1])],
        "phase_distribution": df["phase"].value_counts().to_dict(),
        "feature_strategy": feature_mode,
        "selected_feature_count": len(selected_cols),
        "hormone_feature_count": len(hormone_cols_selected),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "cv_summary": cv_summary,
        "test_summary": test_summary,
        "selected_model": winner,
        "classification_report": cls_report,
        "paths": {
            "phase_predictions": pred_path,
            "phase_confusion_matrix": cm_path,
            "phase_feature_correlation": mi_path,
            "phase_model_artifact": artifact_path,
            "phase_model_comparison": comparison_path,
            "phase_model_tuning_cv": tuning_path,
        },
        "notes": [
            "Label is menstrual phase from merged_women_data.csv",
            "Correlation table uses mutual information with preprocessed features",
            "Split is participant-grouped to reduce leakage",
            "Per-model candidate tuning was performed using GroupKFold on train participants.",
        ],
    }
    report_path = os.path.join(data_dir, "phase_model_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nSaved outputs:")
    print(f"- {pred_path}")
    print(f"- {cm_path}")
    print(f"- {mi_path}")
    print(f"- {artifact_path}")
    print(f"- {comparison_path}")
    print(f"- {tuning_path}")
    print(f"- {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
