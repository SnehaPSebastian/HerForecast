import pandas as pd
import os

# Set working directory
data_dir = os.path.dirname(os.path.abspath(__file__))

# Load all CSV files
print("Loading CSV files...")
wrist_temp = pd.read_csv(os.path.join(data_dir, "wrist_temperature.csv"))
oxygen = pd.read_csv(os.path.join(data_dir, "estimated_oxygen_variation.csv"))
hrv = pd.read_csv(os.path.join(data_dir, "heart_rate_variability_details.csv"))
computed_temp = pd.read_csv(os.path.join(data_dir, "computed_temperature.csv"))
stress = pd.read_csv(os.path.join(data_dir, "stress_score.csv"))
hormones = pd.read_csv(os.path.join(data_dir, "hormones_and_selfreport.csv"))

print(f"  wrist_temperature:              {wrist_temp.shape}")
print(f"  estimated_oxygen_variation:      {oxygen.shape}")
print(f"  heart_rate_variability_details:  {hrv.shape}")
print(f"  computed_temperature:            {computed_temp.shape}")
print(f"  stress_score:                    {stress.shape}")
print(f"  hormones_and_selfreport:         {hormones.shape}")

# Common join keys across all files
common_keys = ["id", "study_interval", "is_weekend", "day_in_study"]

# ---- Step 1: Aggregate the high-frequency data to daily level ----
# This avoids a massive Cartesian product from merging minute-level data.

print("\nAggregating high-frequency data to daily level...")

# Wrist temperature: daily mean of temperature_diff_from_baseline
wrist_temp_daily = (
    wrist_temp.groupby(common_keys, dropna=False)
    .agg(
        wrist_temp_mean=("temperature_diff_from_baseline", "mean"),
        wrist_temp_min=("temperature_diff_from_baseline", "min"),
        wrist_temp_max=("temperature_diff_from_baseline", "max"),
        wrist_temp_std=("temperature_diff_from_baseline", "std"),
        wrist_temp_count=("temperature_diff_from_baseline", "count"),
    )
    .reset_index()
)
print(f"  wrist_temperature (daily):       {wrist_temp_daily.shape}")

# Oxygen variation: daily mean of infrared_to_red_signal_ratio
oxygen_daily = (
    oxygen.groupby(common_keys, dropna=False)
    .agg(
        oxygen_ratio_mean=("infrared_to_red_signal_ratio", "mean"),
        oxygen_ratio_min=("infrared_to_red_signal_ratio", "min"),
        oxygen_ratio_max=("infrared_to_red_signal_ratio", "max"),
        oxygen_ratio_std=("infrared_to_red_signal_ratio", "std"),
        oxygen_ratio_count=("infrared_to_red_signal_ratio", "count"),
    )
    .reset_index()
)
print(f"  estimated_oxygen_variation (daily): {oxygen_daily.shape}")

# HRV: daily aggregates
hrv_daily = (
    hrv.groupby(common_keys, dropna=False)
    .agg(
        rmssd_mean=("rmssd", "mean"),
        rmssd_std=("rmssd", "std"),
        coverage_mean=("coverage", "mean"),
        low_frequency_mean=("low_frequency", "mean"),
        high_frequency_mean=("high_frequency", "mean"),
        hrv_count=("rmssd", "count"),
    )
    .reset_index()
)
print(f"  heart_rate_variability (daily):  {hrv_daily.shape}")

# Stress score: daily aggregates
stress_daily = (
    stress.groupby(common_keys, dropna=False)
    .agg(
        stress_score_mean=("stress_score", "mean"),
        stress_score_max=("stress_score", "max"),
        sleep_points_mean=("sleep_points", "mean"),
        responsiveness_points_mean=("responsiveness_points", "mean"),
        exertion_points_mean=("exertion_points", "mean"),
        stress_count=("stress_score", "count"),
    )
    .reset_index()
)
print(f"  stress_score (daily):            {stress_daily.shape}")

# Computed temperature: already per-sleep-session, aggregate to daily
# Use sleep_start_day_in_study as the day key
computed_temp_renamed = computed_temp.rename(columns={"sleep_start_day_in_study": "day_in_study"})
# Keep only relevant columns for daily merge
ct_keys = ["id", "study_interval", "is_weekend", "day_in_study"]
ct_cols = ["nightly_temperature", "baseline_relative_sample_sum",
           "baseline_relative_sample_sum_of_squares",
           "baseline_relative_nightly_standard_deviation",
           "baseline_relative_sample_standard_deviation"]
computed_temp_daily = (
    computed_temp_renamed.groupby(ct_keys, dropna=False)[ct_cols]
    .mean()
    .reset_index()
)
computed_temp_daily.columns = ct_keys + [
    "nightly_temp_mean",
    "baseline_rel_sample_sum",
    "baseline_rel_sample_sum_sq",
    "baseline_rel_nightly_std",
    "baseline_rel_sample_std",
]
print(f"  computed_temperature (daily):    {computed_temp_daily.shape}")

# Hormones: already daily-level, keep as-is (drop duplicate common keys)
hormones_clean = hormones.copy()
print(f"  hormones_and_selfreport (daily): {hormones_clean.shape}")

# ---- Step 2: Merge all daily-level datasets ----
print("\nMerging all datasets on common keys...")

# Start with hormones as the base (daily-level, richest self-report data)
merged = hormones_clean

# Merge each dataset using outer join to keep all data
for name, df in [
    ("wrist_temp_daily", wrist_temp_daily),
    ("oxygen_daily", oxygen_daily),
    ("hrv_daily", hrv_daily),
    ("stress_daily", stress_daily),
    ("computed_temp_daily", computed_temp_daily),
]:
    before = merged.shape
    merged = pd.merge(merged, df, on=common_keys, how="outer")
    print(f"  After merging {name:30s}: {before} -> {merged.shape}")

# Sort by id and day_in_study
merged = merged.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)

# ---- Step 3: Save ----
output_path = os.path.join(data_dir, "merged_women_data.csv")
merged.to_csv(output_path, index=False)

print(f"\nFinal merged dataset shape: {merged.shape}")
print(f"Columns: {list(merged.columns)}")
print(f"\nSaved to: {output_path}")
print("\nFirst 5 rows preview:")
print(merged.head().to_string())
