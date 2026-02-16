import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

data_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(data_dir, "merged_women_data.csv"))
print(f"Loaded data: {df.shape}")

# ============================================================
# 1. FIX INCONSISTENT VALUES
# ============================================================
numeric_to_ordinal = {"1": "Very Low/Little", "2": "Low", "3": "Moderate", "4": "High", "5": "Very High"}
for col in ["headaches", "stress"]:
    df[col] = df[col].replace(numeric_to_ordinal)
print("Fixed inconsistent categorical values in 'headaches' and 'stress'.")

# ============================================================
# 2. ENCODE ORDINAL CATEGORICAL COLUMNS
# ============================================================
ordinal_scale = {
    "Not at all": 0, "Very Low/Little": 1, "Very Low": 1,
    "Low": 2, "Moderate": 3, "High": 4, "Very High": 5,
}
ordinal_columns = [
    "appetite", "exerciselevel", "headaches", "cramps", "sorebreasts",
    "fatigue", "sleepissue", "moodswing", "stress", "foodcravings",
    "indigestion", "bloating",
]
for col in ordinal_columns:
    df[col] = df[col].map(ordinal_scale)
print(f"Ordinal-encoded {len(ordinal_columns)} symptom columns (0-5).")

flow_volume_scale = {
    "Not at all": 0, "Spotting / Very Light": 1, "Light": 2,
    "Somewhat Light": 3, "Moderate": 4, "Somewhat Heavy": 5,
    "Heavy": 6, "Very Heavy": 7,
}
df["flow_volume"] = df["flow_volume"].map(flow_volume_scale)
print("Ordinal-encoded 'flow_volume' (0-7).")

# ============================================================
# 3. ENCODE NOMINAL CATEGORICAL COLUMNS
# ============================================================
phase_dummies = pd.get_dummies(df["phase"], prefix="phase", dtype=int)
df = pd.concat([df, phase_dummies], axis=1)
df.drop(columns=["phase"], inplace=True)

flow_color_dummies = pd.get_dummies(df["flow_color"], prefix="flow_color", dtype=int)
df = pd.concat([df, flow_color_dummies], axis=1)
df.drop(columns=["flow_color"], inplace=True)

df["is_weekend"] = df["is_weekend"].astype(int)
print("One-hot encoded 'phase' and 'flow_color'; converted 'is_weekend' to 0/1.")

# ============================================================
# 4. HANDLE THE GRANULARITY PROBLEM
# ============================================================
# Different sensors sampled at different rates:
#   - wrist_temperature:    1 per minute  -> max ~1440 readings/day
#   - oxygen_variation:     1 per minute  -> max ~1440 readings/day
#   - HRV:                  1 per 5 min   -> max ~288 readings/day
#   - stress_score:         irregular     -> typically 1-4 readings/day
#   - computed_temperature: 1 per sleep   -> 1 reading/day
#   - hormones/selfreport:  1 per day     -> 1 reading/day
#
# Problem: a "mean" from 1440 readings is far more reliable than
# a "mean" from 2 readings. Raw counts are not comparable.
#
# Solution: Convert raw counts into a COVERAGE RATIO (0-1) that
# represents what fraction of expected daily readings were captured.
# This makes all coverage columns directly comparable regardless
# of the original sampling rate.

expected_daily_samples = {
    "wrist_temp_count":    1440,   # 1 per minute, 24 hours
    "oxygen_ratio_count":  1440,   # 1 per minute, 24 hours
    "hrv_count":           288,    # 1 per 5 minutes, 24 hours
    "stress_count":        4,      # typically ~4 readings per day
}

print("\nConverting raw sample counts to coverage ratios (0-1):")
for count_col, expected in expected_daily_samples.items():
    coverage_col = count_col.replace("_count", "_coverage")
    df[coverage_col] = (df[count_col] / expected).clip(upper=1.0)
    print(f"  {count_col} (max {expected}/day) -> {coverage_col}")

# Drop the original raw count columns (they'd bias the model)
raw_count_cols = list(expected_daily_samples.keys())
df.drop(columns=raw_count_cols, inplace=True)
print(f"Dropped raw count columns: {raw_count_cols}")

# ============================================================
# 5. PER-PARTICIPANT Z-SCORE NORMALIZATION
# ============================================================
# Each participant has their own baseline physiology. A wrist temp
# of 34.5°C might be high for one person and low for another.
# Z-scoring within each participant removes individual baselines
# so the model learns relative changes, not absolute levels.
#
# This also solves the cross-sensor scale problem: after z-scoring,
# ALL sensor features are on the same scale (mean=0, std=1) regardless
# of whether the original was temperature (°C) or HRV (ms).

# Columns to z-score per participant (sensor + hormone data)
sensor_and_hormone_cols = [
    # Wrist temperature
    "wrist_temp_mean", "wrist_temp_min", "wrist_temp_max", "wrist_temp_std",
    # Oxygen
    "oxygen_ratio_mean", "oxygen_ratio_min", "oxygen_ratio_max", "oxygen_ratio_std",
    # HRV
    "rmssd_mean", "rmssd_std", "coverage_mean", "low_frequency_mean", "high_frequency_mean",
    # Stress
    "stress_score_mean", "stress_score_max",
    "sleep_points_mean", "responsiveness_points_mean", "exertion_points_mean",
    # Computed temperature
    "nightly_temp_mean", "baseline_rel_sample_sum", "baseline_rel_sample_sum_sq",
    "baseline_rel_nightly_std", "baseline_rel_sample_std",
    # Hormones
    "lh", "estrogen", "pdg",
]

print(f"\nApplying per-participant z-score normalization to {len(sensor_and_hormone_cols)} columns...")

for col in sensor_and_hormone_cols:
    # Group by participant (id) and z-score within each group
    grouped = df.groupby("id")[col]
    df[col] = grouped.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

print("  Per-participant z-scoring complete.")
print("  -> Each participant's features now have mean~0, std~1")
print("  -> Cross-sensor scales are now comparable")

# ============================================================
# 6. SCALE REMAINING COLUMNS (Global MinMax for ordinals & coverage)
# ============================================================
# Ordinal symptom columns (already 0-5 or 0-7) and coverage ratios
# (already 0-1) just need to be rescaled to 0-1 for consistency.

ordinal_and_coverage = ordinal_columns + ["flow_volume"] + [
    "wrist_temp_coverage", "oxygen_ratio_coverage",
    "hrv_coverage", "stress_coverage",
]

# Also scale day_in_study
other_to_scale = ["day_in_study"]

cols_to_minmax = ordinal_and_coverage + other_to_scale
print(f"\nMinMax scaling {len(cols_to_minmax)} ordinal/coverage/time columns to 0-1.")

for col in cols_to_minmax:
    cmin, cmax = df[col].min(), df[col].max()
    if cmax > cmin:
        df[col] = (df[col] - cmin) / (cmax - cmin)
    else:
        df[col] = 0

# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================
missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]
print(f"\nColumns with missing values ({len(missing_cols)}):")
for col, count in missing_cols.items():
    pct = count / len(df) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")

# For z-scored sensor columns: fill with 0 (= participant's own mean)
# This is the most neutral imputation for z-scored data.
for col in sensor_and_hormone_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0.0)

# For ordinal/coverage columns: fill with median
remaining_missing_cols = df.columns[df.isnull().any()].tolist()
for col in remaining_missing_cols:
    df[col] = df[col].fillna(df[col].median())

remaining = df.isnull().sum().sum()
print(f"\nMissing value imputation:")
print(f"  Z-scored sensor/hormone cols -> filled with 0 (= participant mean)")
print(f"  Ordinal/coverage cols -> filled with median")
print(f"  Remaining missing values: {remaining}")

# ============================================================
# 8. SAVE
# ============================================================
output_path = os.path.join(data_dir, "normalized_women_data.csv")
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"FINAL NORMALIZED DATASET")
print(f"{'='*60}")
print(f"Shape: {df.shape}")
print(f"Saved to: {output_path}")

# Show a summary of all feature groups and their value ranges
print(f"\n--- Feature Groups ---")
print(f"\nZ-scored sensor/hormone features ({len(sensor_and_hormone_cols)} cols):")
z_desc = df[sensor_and_hormone_cols].describe().loc[["mean", "std", "min", "max"]].round(3)
print(f"  Mean range:  [{z_desc.loc['mean'].min():.3f}, {z_desc.loc['mean'].max():.3f}]  (should be ~0)")
print(f"  Std range:   [{z_desc.loc['std'].min():.3f}, {z_desc.loc['std'].max():.3f}]  (should be ~1)")

print(f"\nOrdinal symptom features ({len(ordinal_columns) + 1} cols): all scaled 0-1")
print(f"\nCoverage ratio features (4 cols): all 0-1")
print(f"\nOne-hot features ({len(phase_dummies.columns) + len(flow_color_dummies.columns)} cols): all 0/1")
print(f"\nIdentifiers kept as-is: id, study_interval")

print(f"\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")
