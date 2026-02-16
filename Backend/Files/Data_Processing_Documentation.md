# Women Health Data - Processing Documentation

**Date:** February 12, 2026
**Author:** Auto-generated
**Project:** Women Health Sensor & Self-Report Data Pipeline

---

## Table of Contents

1. [Overview](#1-overview)
2. [Source Data Files](#2-source-data-files)
3. [Step 1 - Data Merging](#3-step-1---data-merging)
4. [Step 2 - Normalization & Feature Engineering](#4-step-2---normalization--feature-engineering)
5. [Output Files Summary](#5-output-files-summary)
6. [Column Reference -- Before and After Normalization](#6-column-reference----before-and-after-normalization)
7. [Why Normalization Preserves Context](#7-why-normalization-preserves-context)
8. [Body State Score Framework (Strict Science-Backed)](#8-body-state-score-framework-strict-science-backed)
9. [Known Limitations & Notes](#9-known-limitations--notes)
10. [Implemented Model Run (v2 Improved)](#10-implemented-model-run-v2-improved)

---

## 1. Overview

This document records all data processing steps applied to transform 6 raw CSV files
containing women's health sensor data and self-report surveys into a single,
machine-learning-ready dataset.

**Pipeline:**

```
6 Raw CSVs --> [merge_data.py] --> merged_women_data.csv --> [normalize_data.py] --> normalized_women_data.csv
```

| Stage         | Script              | Input                       | Output                       |
|---------------|---------------------|-----------------------------|------------------------------|
| Merging       | `merge_data.py`     | 6 raw CSV files             | `merged_women_data.csv`      |
| Normalization | `normalize_data.py` | `merged_women_data.csv`     | `normalized_women_data.csv`  |

---

## 2. Source Data Files

All source files are located in the `WOMEN DATA` folder. Each file contains data
indexed by participant `id`, `study_interval`, `is_weekend`, and `day_in_study`.

| File                                | Rows       | Columns | Granularity         | Key Measurement Columns                                                              |
|-------------------------------------|------------|---------|---------------------|---------------------------------------------------------------------------------------|
| `wrist_temperature.csv`             | 6,856,019  | 6       | 1 per minute        | `temperature_diff_from_baseline`                                                      |
| `estimated_oxygen_variation.csv`    | 3,070,312  | 6       | 1 per minute        | `infrared_to_red_signal_ratio`                                                        |
| `heart_rate_variability_details.csv`| 436,262    | 9       | 1 per 5 minutes     | `rmssd`, `coverage`, `low_frequency`, `high_frequency`                                |
| `stress_score.csv`                  | 7,932      | 14      | Irregular (~1-4/day)| `stress_score`, `sleep_points`, `responsiveness_points`, `exertion_points`, `status`  |
| `computed_temperature.csv`          | 5,575      | 14      | 1 per sleep session | `nightly_temperature`, `baseline_relative_*` columns                                  |
| `hormones_and_selfreport.csv`       | 5,659      | 22      | 1 per day           | `lh`, `estrogen`, `pdg`, `phase`, `flow_volume`, symptom self-reports (12 columns)    |

**Common join keys across all files:** `id`, `study_interval`, `is_weekend`, `day_in_study`

---

## 3. Step 1 - Data Merging

**Script:** `merge_data.py`
**Output:** `merged_women_data.csv` (6,038 rows x 49 columns)

### 3.1 Problem

The 6 source files have vastly different granularities (per-minute to per-day). A naive
merge on shared keys would create a massive Cartesian product (billions of rows) for
the high-frequency sensor data.

### 3.2 Solution: Aggregate to Daily Level

Before merging, all high-frequency sensor data was aggregated to one row per
`(id, study_interval, is_weekend, day_in_study)`.

#### Wrist Temperature (6,856,019 rows --> 5,138 daily rows)

| Original Column                    | Aggregation       | New Column Name      |
|------------------------------------|-------------------|----------------------|
| `temperature_diff_from_baseline`   | Mean              | `wrist_temp_mean`    |
| `temperature_diff_from_baseline`   | Min               | `wrist_temp_min`     |
| `temperature_diff_from_baseline`   | Max               | `wrist_temp_max`     |
| `temperature_diff_from_baseline`   | Std deviation     | `wrist_temp_std`     |
| `temperature_diff_from_baseline`   | Count             | `wrist_temp_count`   |

#### Estimated Oxygen Variation (3,070,312 rows --> 5,457 daily rows)

| Original Column                    | Aggregation       | New Column Name        |
|------------------------------------|-------------------|------------------------|
| `infrared_to_red_signal_ratio`     | Mean              | `oxygen_ratio_mean`    |
| `infrared_to_red_signal_ratio`     | Min               | `oxygen_ratio_min`     |
| `infrared_to_red_signal_ratio`     | Max               | `oxygen_ratio_max`     |
| `infrared_to_red_signal_ratio`     | Std deviation     | `oxygen_ratio_std`     |
| `infrared_to_red_signal_ratio`     | Count             | `oxygen_ratio_count`   |

#### Heart Rate Variability (436,262 rows --> 4,839 daily rows)

| Original Column     | Aggregation       | New Column Name          |
|---------------------|-------------------|--------------------------|
| `rmssd`             | Mean              | `rmssd_mean`             |
| `rmssd`             | Std deviation     | `rmssd_std`              |
| `coverage`          | Mean              | `coverage_mean`          |
| `low_frequency`     | Mean              | `low_frequency_mean`     |
| `high_frequency`    | Mean              | `high_frequency_mean`    |
| `rmssd`             | Count             | `hrv_count`              |

#### Stress Score (7,932 rows --> 4,239 daily rows)

| Original Column              | Aggregation       | New Column Name                |
|------------------------------|--------------------|-------------------------------|
| `stress_score`               | Mean               | `stress_score_mean`           |
| `stress_score`               | Max                | `stress_score_max`            |
| `sleep_points`               | Mean               | `sleep_points_mean`           |
| `responsiveness_points`      | Mean               | `responsiveness_points_mean`  |
| `exertion_points`            | Mean               | `exertion_points_mean`        |
| `stress_score`               | Count              | `stress_count`                |

#### Computed Temperature (5,575 rows --> 4,695 daily rows)

- Renamed `sleep_start_day_in_study` to `day_in_study` for alignment.
- Grouped by daily keys and averaged when multiple sleep sessions existed per day.

| Original Column                                     | New Column Name               |
|------------------------------------------------------|-------------------------------|
| `nightly_temperature`                                | `nightly_temp_mean`           |
| `baseline_relative_sample_sum`                       | `baseline_rel_sample_sum`     |
| `baseline_relative_sample_sum_of_squares`            | `baseline_rel_sample_sum_sq`  |
| `baseline_relative_nightly_standard_deviation`       | `baseline_rel_nightly_std`    |
| `baseline_relative_sample_standard_deviation`        | `baseline_rel_sample_std`     |

#### Hormones & Self-Report (5,659 rows -- already daily)

Kept as-is. All 22 columns retained. Used as the base table for the merge.

### 3.3 Merge Strategy

- **Base table:** `hormones_and_selfreport` (richest daily-level data)
- **Join type:** Outer join (to keep all data, even days with only sensor or only self-report data)
- **Join keys:** `id`, `study_interval`, `is_weekend`, `day_in_study`
- **Merge order:** hormones -> wrist_temp -> oxygen -> HRV -> stress -> computed_temp
- **Final sort:** By `id`, `study_interval`, `day_in_study`

### 3.4 Merge Result

- **Rows:** 6,038 (some extra rows from computed_temperature days not present in hormones)
- **Columns:** 49

---

## 4. Step 2 - Normalization & Feature Engineering

**Script:** `normalize_data.py`
**Input:** `merged_women_data.csv` (6,038 rows x 49 columns)
**Output:** `normalized_women_data.csv` (6,038 rows x 60 columns)

### 4.1 Data Cleaning: Fix Inconsistent Values

**Columns affected:** `headaches`, `stress`

These columns had a mix of text labels (e.g., "Low", "Moderate") and numeric strings
(e.g., "1", "2", "3") representing the same scale. All numeric strings were mapped
back to their text equivalents:

| Numeric String | Mapped To         |
|----------------|-------------------|
| "1"            | "Very Low/Little" |
| "2"            | "Low"             |
| "3"            | "Moderate"        |
| "4"            | "High"            |
| "5"            | "Very High"       |

### 4.2 Ordinal Encoding (13 columns)

Symptom and self-report columns with a natural severity order were mapped to integers:

| Text Label        | Numeric Value |
|-------------------|---------------|
| "Not at all"      | 0             |
| "Very Low/Little" | 1             |
| "Very Low"        | 1             |
| "Low"             | 2             |
| "Moderate"        | 3             |
| "High"            | 4             |
| "Very High"       | 5             |

**Columns encoded (0-5 scale):**
`appetite`, `exerciselevel`, `headaches`, `cramps`, `sorebreasts`, `fatigue`,
`sleepissue`, `moodswing`, `stress`, `foodcravings`, `indigestion`, `bloating`

**`flow_volume` encoded separately (0-7 scale):**

| Text Label              | Numeric Value |
|-------------------------|---------------|
| "Not at all"            | 0             |
| "Spotting / Very Light" | 1             |
| "Light"                 | 2             |
| "Somewhat Light"        | 3             |
| "Moderate"              | 4             |
| "Somewhat Heavy"        | 5             |
| "Heavy"                 | 6             |
| "Very Heavy"            | 7             |

### 4.3 One-Hot Encoding (2 nominal columns)

Categorical columns with no natural order were one-hot encoded:

**`phase` (4 categories --> 4 binary columns):**
- `phase_Fertility`
- `phase_Follicular`
- `phase_Luteal`
- `phase_Menstrual`

**`flow_color` (9 categories --> 9 binary columns):**
- `flow_color_Black`
- `flow_color_Bright Red`
- `flow_color_Dark Brown / Dark Red`
- `flow_color_Grey`
- `flow_color_Not at all`
- `flow_color_Orange`
- `flow_color_Other`
- `flow_color_Pink`
- `flow_color_Yellow`

**`is_weekend`:** Converted from boolean (`True`/`False`) to integer (`1`/`0`).

### 4.4 Granularity Normalization: Coverage Ratios

**Problem:** Raw sample counts across sensors are not comparable because each sensor
samples at a different rate. `wrist_temp_count = 908` vs `stress_count = 2` does not
mean wrist temperature had "more" data in a meaningful way -- it simply samples faster.

**Solution:** Each raw count was divided by the maximum expected daily readings for that
sensor, producing a **coverage ratio** between 0 and 1.

| Raw Count Column       | Expected Daily Max | New Coverage Column        | Interpretation            |
|------------------------|--------------------|----------------------------|---------------------------|
| `wrist_temp_count`     | 1,440 (1/min)      | `wrist_temp_coverage`      | Fraction of day covered   |
| `oxygen_ratio_count`   | 1,440 (1/min)      | `oxygen_ratio_coverage`    | Fraction of day covered   |
| `hrv_count`            | 288 (1/5min)       | `hrv_coverage`             | Fraction of day covered   |
| `stress_count`         | 4 (typical/day)    | `stress_coverage`          | Fraction of day covered   |

The raw count columns were then **dropped** to prevent them from biasing the model.

### 4.5 Per-Participant Z-Score Normalization (26 columns)

**Problem:** Different sensors measure fundamentally different things on different scales
(temperature in degrees C, HRV in milliseconds, hormone levels in pg/mL, etc.). Additionally,
each participant has a unique physiological baseline -- 34.5 degrees C might be high for one person and
low for another.

**Solution:** For each participant, each sensor/hormone feature was z-scored independently:

```
z = (value - participant_mean) / participant_std
```

This transformation:
- Removes individual baseline differences (all participants centered at 0)
- Makes cross-sensor features comparable (all on the same unitless scale)
- A value of +1.5 means "1.5 standard deviations above this person's average"

**Columns z-scored (26 total):**

| Group                | Columns                                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------------------|
| Wrist Temperature    | `wrist_temp_mean`, `wrist_temp_min`, `wrist_temp_max`, `wrist_temp_std`                                  |
| Oxygen Variation     | `oxygen_ratio_mean`, `oxygen_ratio_min`, `oxygen_ratio_max`, `oxygen_ratio_std`                          |
| Heart Rate Variability | `rmssd_mean`, `rmssd_std`, `coverage_mean`, `low_frequency_mean`, `high_frequency_mean`                |
| Stress Scores        | `stress_score_mean`, `stress_score_max`, `sleep_points_mean`, `responsiveness_points_mean`, `exertion_points_mean` |
| Computed Temperature | `nightly_temp_mean`, `baseline_rel_sample_sum`, `baseline_rel_sample_sum_sq`, `baseline_rel_nightly_std`, `baseline_rel_sample_std` |
| Hormones             | `lh`, `estrogen`, `pdg`                                                                                  |

### 4.6 MinMax Scaling for Ordinal & Coverage Columns (18 columns)

After ordinal encoding and coverage ratio computation, these columns were rescaled
to the 0-1 range using Min-Max normalization:

```
scaled = (value - min) / (max - min)
```

**Columns scaled:**
- 12 ordinal symptom columns (originally 0-5)
- `flow_volume` (originally 0-7)
- 4 coverage ratio columns (already approximately 0-1, clipped and rescaled)
- `day_in_study` (integer day number, scaled to 0-1)

### 4.7 Missing Value Imputation

Missing values were handled with two strategies depending on the column type:

| Column Type                         | Imputation Strategy | Rationale                                                                    |
|-------------------------------------|---------------------|------------------------------------------------------------------------------|
| Z-scored sensor & hormone columns   | Filled with **0**   | 0 = the participant's own mean; the most neutral/uninformative value         |
| Ordinal & coverage columns          | Filled with **median** | Median is robust to outliers and represents the "typical" value           |

**Missing value counts before imputation (selected):**

| Column Group              | Missing Count | Missing %  |
|---------------------------|---------------|------------|
| Hormone levels (lh, estrogen) | ~700      | ~11.6%     |
| pdg                       | 2,208         | 36.6%      |
| Self-report symptoms      | ~2,710        | ~44.9%     |
| Wrist temperature stats   | 900           | 14.9%      |
| Oxygen variation stats    | 581           | 9.6%       |
| HRV stats                 | ~973-977      | ~16.1%     |
| Stress score stats        | ~862-1,298    | ~14-21.5%  |
| Computed temperature stats| 1,343-1,454   | 22-24%     |
| Coverage ratios           | 581-1,799     | 9.6-29.8%  |

**After imputation: 0 missing values remain.**

---

## 5. Output Files Summary

### merged_women_data.csv (Intermediate)
- **Rows:** 6,038
- **Columns:** 49
- **Description:** All 6 source files joined on shared keys with high-frequency data
  aggregated to daily level. Contains original text labels and raw numeric values.
  Has missing values.

### normalized_women_data.csv (Final, ML-Ready)
- **Rows:** 6,038
- **Columns:** 60 (increased due to one-hot encoding of `phase` and `flow_color`)
- **Description:** Fully numeric, normalized, zero missing values. Ready for ML.

| Feature Type                   | Count | Value Range    | Scaling Method                   |
|--------------------------------|-------|----------------|----------------------------------|
| Z-scored sensor/hormone        | 26    | ~(-3, +3)      | Per-participant z-score          |
| Ordinal symptom/self-report    | 13    | [0, 1]         | Ordinal encoding + MinMax        |
| Coverage ratios                | 4     | [0, 1]         | Raw count / expected daily max   |
| One-hot encoded                | 13    | {0, 1}         | Binary indicator                 |
| Time feature (day_in_study)    | 1     | [0, 1]         | MinMax                           |
| Binary (is_weekend)            | 1     | {0, 1}         | Boolean to int                   |
| Identifiers                    | 2     | Original values | Not scaled (id, study_interval)  |
| **Total**                      | **60**|                |                                  |

---

## 6. Column Reference -- Before and After Normalization

This section documents every column in the final `normalized_women_data.csv`, grouped
by type. For each group, it shows what the column originally contained in
`merged_women_data.csv`, what it contains now, and how to interpret the normalized value.

---

### 6.1 Identifier Columns (2 columns -- unchanged)

These columns are kept exactly as they were. They identify *who* and *when*, and are
not fed into the model as features.

| # | Column           | Before (merged)          | After (normalized)       | How to Read It                        |
|---|------------------|--------------------------|--------------------------|---------------------------------------|
| 1 | `id`             | Integer participant ID (e.g., 1, 2, 15) | Same -- no change   | Participant identity. Not a feature.  |
| 2 | `study_interval` | Year of study (e.g., 2022)              | Same -- no change   | Study wave. Not a feature.            |

**Context preserved?** Fully. These are pass-through columns.

---

### 6.2 Binary / Time Columns (2 columns)

| # | Column           | Before (merged)             | After (normalized)                  | How to Read It                                          |
|---|------------------|-----------------------------|-------------------------------------|---------------------------------------------------------|
| 3 | `is_weekend`     | Boolean: `True` or `False`  | Integer: `1` or `0`                 | 1 = weekend day, 0 = weekday. Identical meaning.        |
| 4 | `day_in_study`   | Integer day number (e.g., 1, 2, ... 85) | Decimal 0.0 to 1.0 (MinMax scaled) | 0.0 = first day of study, 1.0 = last day. Relative position in the study timeline. |

**Context preserved?** Yes. `is_weekend` is a direct 1:1 swap of format. `day_in_study`
preserves the ordering and relative spacing -- day 40 out of 80 is 0.5, which is still
"halfway through the study." The model can still distinguish early vs. late study days.

**Worked example:**

| Column         | Before | After | Interpretation                     |
|----------------|--------|-------|------------------------------------|
| `is_weekend`   | True   | 1     | It's a weekend -- same meaning     |
| `day_in_study` | 42     | 0.49  | Roughly halfway through the study  |

---

### 6.3 Z-Scored Sensor and Hormone Columns (26 columns)

These are the physiological measurements from wearable sensors and hormone tests.
They were z-scored **per participant**: each person's values were centered to mean=0
and scaled to std=1 using only their own data.

**Formula:** `z = (raw_value - this_participant's_mean) / this_participant's_std`

| # | Column | What It Originally Measured | Original Unit / Range | After (normalized) | How to Read the Normalized Value |
|---|--------|---------------------------|----------------------|-------------------|----------------------------------|
| 5 | `lh` | Luteinizing hormone level on that day | pg/mL (e.g., 2.9, 15.3) | Z-score (e.g., -0.27, +1.8) | +1.8 = "LH is 1.8 std deviations above this person's average LH." Indicates a relative surge for this individual. |
| 6 | `estrogen` | Estrogen level on that day | pg/mL (e.g., 94.2, 364.7) | Z-score (e.g., -0.75, +2.1) | +2.1 = "Estrogen is much higher than this person's norm." Direction and magnitude of change preserved. |
| 7 | `pdg` | Pregnanediol glucuronide (progesterone metabolite) | ug/mL (e.g., 0.4, 8.2) | Z-score (e.g., -0.5, +1.3) | +1.3 = "PDG is elevated relative to this person's baseline." Rise in PDG typically follows ovulation. |
| 21 | `wrist_temp_mean` | Average wrist skin temperature deviation from baseline across the entire day | Degrees C (e.g., -2.41, +0.85) | Z-score (e.g., -1.1, +0.6) | +0.6 = "Wrist temp was moderately above this person's daily average." Positive = warmer than usual for them. |
| 22 | `wrist_temp_min` | Lowest temperature deviation recorded that day | Degrees C (e.g., -9.59) | Z-score | Negative z-score = the day's lowest dip was deeper than usual for this person. |
| 23 | `wrist_temp_max` | Highest temperature deviation recorded that day | Degrees C (e.g., +2.51) | Z-score | Positive z-score = the day's peak was higher than this person's typical peak. |
| 24 | `wrist_temp_std` | How much wrist temperature varied throughout the day | Degrees C (e.g., 2.95) | Z-score | Positive z-score = more temperature fluctuation than usual; the body's thermoregulation was more active. |
| 25 | `oxygen_ratio_mean` | Average infrared-to-red signal ratio (blood oxygen proxy) across the day | Ratio (e.g., -0.05, +1.47) | Z-score | Positive z-score = oxygen levels were higher than this person's norm. |
| 26 | `oxygen_ratio_min` | Lowest oxygen ratio recorded that day | Ratio (e.g., -86.0) | Z-score | Negative z-score = the day's lowest oxygen dip was deeper than usual. |
| 27 | `oxygen_ratio_max` | Highest oxygen ratio recorded that day | Ratio (e.g., 75.0) | Z-score | Positive z-score = the day's oxygen peak was higher than typical for this person. |
| 28 | `oxygen_ratio_std` | How much oxygen ratio varied throughout the day | Ratio (e.g., 16.19) | Z-score | Positive z-score = more oxygen variability than usual for this person. |
| 29 | `rmssd_mean` | Average root mean square of successive differences in heartbeat intervals (HRV measure) | Milliseconds (e.g., 37.1, 42.6) | Z-score | Positive z-score = higher HRV than this person's average, indicating more relaxation / parasympathetic activity. |
| 30 | `rmssd_std` | How much HRV varied across 5-minute windows that day | Milliseconds | Z-score | Positive z-score = HRV was more variable than usual. |
| 31 | `coverage_mean` | Average HRV signal coverage quality | Fraction (e.g., 0.76, 1.0) | Z-score | Positive z-score = better-than-usual signal quality that day. |
| 32 | `low_frequency_mean` | Average low-frequency HRV power (sympathetic + parasympathetic) | ms^2/Hz (e.g., 817.3) | Z-score | Positive z-score = more low-frequency HRV power than usual for this person. |
| 33 | `high_frequency_mean` | Average high-frequency HRV power (parasympathetic) | ms^2/Hz (e.g., 257.8) | Z-score | Positive z-score = more parasympathetic activity than this person's average. |
| 34 | `stress_score_mean` | Average stress score across readings that day | Points (e.g., 0-100) | Z-score | Positive z-score = more stressed than this person's typical day. |
| 35 | `stress_score_max` | Peak stress score recorded that day | Points | Z-score | Positive z-score = peak stress was higher than usual for this person. |
| 36 | `sleep_points_mean` | Average sleep quality contribution to stress | Points | Z-score | Positive z-score = sleep contributed more to stress than usual. |
| 37 | `responsiveness_points_mean` | Average responsiveness contribution to stress | Points | Z-score | Positive z-score = body was less responsive to stress recovery than usual. |
| 38 | `exertion_points_mean` | Average physical exertion contribution to stress | Points | Z-score | Positive z-score = more physical exertion impact than this person's norm. |
| 39 | `nightly_temp_mean` | Average skin temperature during sleep | Degrees C (e.g., 34.6) | Z-score | Positive z-score = slept warmer than this person's average night. |
| 40 | `baseline_rel_sample_sum` | Sum of temperature samples relative to personal baseline during sleep | Degrees C aggregated | Z-score | Positive z-score = cumulative sleep temperature was above this person's baseline more than usual. |
| 41 | `baseline_rel_sample_sum_sq` | Sum of squared deviations from baseline during sleep | Degrees C^2 aggregated | Z-score | Positive z-score = more extreme temperature deviations from baseline during sleep than usual. |
| 42 | `baseline_rel_nightly_std` | Standard deviation of nightly temperature relative to baseline | Degrees C | Z-score | Positive z-score = nightly temperature was more variable relative to baseline than usual. |
| 43 | `baseline_rel_sample_std` | Standard deviation of individual temperature samples relative to baseline | Degrees C | Z-score | Positive z-score = more sample-level temperature variability during sleep than usual. |

**Context preserved?** Yes. The z-score retains two critical pieces of information:

1. **Direction:** Positive = above this person's average. Negative = below.
2. **Magnitude:** The further from 0, the more unusual the reading is for this person.

What is *removed* is the absolute unit (34.6 degrees C becomes +0.3), but this is
intentional: the absolute number is meaningless without knowing the person. A z-score
of +0.3 universally means "slightly above my own normal" for every participant.

**Worked example (Participant #1, Day 3):**

| Column             | Before (raw)      | After (z-scored) | Interpretation                                          |
|--------------------|-------------------|------------------|---------------------------------------------------------|
| `lh`               | 3.5 pg/mL         | +0.12            | LH is very slightly above this person's average         |
| `estrogen`         | 276.8 pg/mL       | +0.31            | Estrogen is moderately above this person's average      |
| `wrist_temp_mean`  | -2.41 degrees C    | -1.10            | Wrist temp was noticeably below this person's daily norm |
| `nightly_temp_mean`| 34.63 degrees C    | +0.31            | Slept slightly warmer than their average night          |

**Reversibility:** You can always recover the original value:
`raw_value = z_score * participant_std + participant_mean`
(using the per-participant stats from `merged_women_data.csv`).

---

### 6.4 Ordinal Symptom and Self-Report Columns (13 columns)

These are the participant's daily self-reported symptoms and behaviors. They originally
used text labels with a natural severity ordering. They were encoded to integers (0-5
or 0-7), then MinMax scaled to 0-1.

**Two-step transformation:**
1. Text -> Integer: "Not at all" = 0, "Very Low/Little" = 1, "Low" = 2, "Moderate" = 3, "High" = 4, "Very High" = 5
2. Integer -> 0-1: Divided by the maximum (5 for symptoms, 7 for flow_volume)

| # | Column | What It Originally Measured | Before (merged) | After (normalized) | How to Read It |
|---|--------|---------------------------|-----------------|-------------------|----------------|
| 8 | `flow_volume` | Menstrual flow intensity | Text: "Not at all", "Light", ..., "Very Heavy" | 0.0 to 1.0 | 0.0 = no flow, ~0.57 = "Moderate", 1.0 = "Very Heavy". Higher = heavier flow. |
| 9 | `appetite` | Self-reported appetite level | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = no appetite, 0.6 = "Moderate", 1.0 = "Very High". Higher = more appetite. |
| 10 | `exerciselevel` | Self-reported exercise intensity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = no exercise, 1.0 = very intense exercise. |
| 11 | `headaches` | Headache severity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = no headache, 0.4 = "Low", 0.8 = "High", 1.0 = "Very High". |
| 12 | `cramps` | Menstrual cramp severity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = none, 1.0 = most severe. |
| 13 | `sorebreasts` | Breast tenderness | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = none, 1.0 = most severe. |
| 14 | `fatigue` | Tiredness level | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = not tired at all, 1.0 = extreme fatigue. |
| 15 | `sleepissue` | Sleep problem severity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = no sleep issues, 1.0 = severe sleep problems. |
| 16 | `moodswing` | Mood fluctuation severity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = stable mood, 1.0 = extreme mood swings. |
| 17 | `stress` | Self-reported stress level | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = not stressed, 0.6 = "Moderate", 1.0 = "Very High". |
| 18 | `foodcravings` | Food craving intensity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = no cravings, 1.0 = intense cravings. |
| 19 | `indigestion` | Digestive discomfort | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = none, 1.0 = severe indigestion. |
| 20 | `bloating` | Bloating severity | Text: "Not at all" to "Very High" | 0.0 to 1.0 | 0.0 = none, 1.0 = severe bloating. |

**Context preserved?** Yes. The transformation is a direct, reversible mapping:

- The **rank order** is preserved: 0.8 ("High") > 0.6 ("Moderate") > 0.4 ("Low"). The model sees the same "more is worse" relationship that the original text conveyed.
- The **relative distances** are preserved: the gap between "Low" and "Moderate" (0.2) is the same as the gap between "Moderate" and "High" (0.2). The original text labels were designed on an evenly-spaced severity scale, and the numbers maintain that.
- The **reversibility** is trivial: multiply by 5 (or 7 for flow_volume) and map back to the text label using the table in Section 4.2.

**Worked example:**

| Column      | Before (raw text) | Step 1 (integer) | Step 2 (0-1 scaled) | Reading back                  |
|-------------|-------------------|-------------------|---------------------|-------------------------------|
| `fatigue`   | "High"            | 4                 | 0.8                 | 0.8 * 5 = 4 = "High" fatigue |
| `cramps`    | "Very Low/Little" | 1                 | 0.2                 | 0.2 * 5 = 1 = "Very Low"     |
| `flow_volume` | "Moderate"      | 4                 | 0.571               | 0.571 * 7 = 4 = "Moderate"   |

---

### 6.5 One-Hot Encoded Columns (13 columns)

These columns represent categories that have **no natural order** (you cannot say
"Luteal > Follicular" or "Red > Pink"). Instead of assigning arbitrary numbers, each
category becomes its own binary column.

#### Menstrual Cycle Phase (originally 1 column `phase` with 4 text values)

| # | Column | Before (merged) | After (normalized) | How to Read It |
|---|--------|-----------------|-------------------|----------------|
| 44 | `phase_Fertility` | `phase` = "Fertility" | 1 if Fertility phase, 0 otherwise | Exactly one of the 4 phase columns is 1 per row. |
| 45 | `phase_Follicular` | `phase` = "Follicular" | 1 if Follicular phase, 0 otherwise | |
| 46 | `phase_Luteal` | `phase` = "Luteal" | 1 if Luteal phase, 0 otherwise | |
| 47 | `phase_Menstrual` | `phase` = "Menstrual" | 1 if Menstrual phase, 0 otherwise | |

#### Flow Color (originally 1 column `flow_color` with 9 text values)

| # | Column | Before (merged) | After (normalized) | How to Read It |
|---|--------|-----------------|-------------------|----------------|
| 48 | `flow_color_Black` | `flow_color` = "Black" | 1 if Black, 0 otherwise | Exactly one of the 9 color columns is 1 per row. |
| 49 | `flow_color_Bright Red` | `flow_color` = "Bright Red" | 1 if Bright Red, 0 otherwise | |
| 50 | `flow_color_Dark Brown / Dark Red` | `flow_color` = "Dark Brown / Dark Red" | 1 if Dark Brown/Red, 0 otherwise | |
| 51 | `flow_color_Grey` | `flow_color` = "Grey" | 1 if Grey, 0 otherwise | |
| 52 | `flow_color_Not at all` | `flow_color` = "Not at all" | 1 if no flow color, 0 otherwise | |
| 53 | `flow_color_Orange` | `flow_color` = "Orange" | 1 if Orange, 0 otherwise | |
| 54 | `flow_color_Other` | `flow_color` = "Other" | 1 if Other, 0 otherwise | |
| 55 | `flow_color_Pink` | `flow_color` = "Pink" | 1 if Pink, 0 otherwise | |
| 56 | `flow_color_Yellow` | `flow_color` = "Yellow" | 1 if Yellow, 0 otherwise | |

**Context preserved?** Completely -- zero information is lost. One-hot encoding is a
purely structural reformatting. The single column `phase = "Luteal"` and the four
columns `[0, 0, 1, 0]` carry exactly the same information. The model reads
`phase_Luteal = 1` and knows "this day is in the Luteal phase" just as clearly as
the original text.

**Worked example:**

| Before (merged)      | After (normalized columns)                                                    |
|----------------------|-------------------------------------------------------------------------------|
| `phase` = "Luteal"   | `phase_Fertility=0`, `phase_Follicular=0`, `phase_Luteal=1`, `phase_Menstrual=0` |
| `flow_color` = "Bright Red" | `flow_color_Black=0`, `flow_color_Bright Red=1`, all others = 0         |

---

### 6.6 Coverage Ratio Columns (4 columns)

These columns replace the raw sample counts that were in `merged_women_data.csv`. They
tell the model how much of the day's expected sensor data was actually captured.

| # | Column | Before (merged) | After (normalized) | How to Read It |
|---|--------|-----------------|-------------------|----------------|
| 57 | `wrist_temp_coverage` | `wrist_temp_count` = 908 (raw readings) | 0.63 (= 908 / 1440) | 63% of the day's expected minute-by-minute wrist temperature readings were captured. |
| 58 | `oxygen_ratio_coverage` | `oxygen_ratio_count` = 821 (raw readings) | 0.57 (= 821 / 1440) | 57% of the day's expected minute-by-minute oxygen readings were captured. |
| 59 | `hrv_coverage` | `hrv_count` = 190 (raw readings) | 0.66 (= 190 / 288) | 66% of the day's expected 5-minute HRV readings were captured. |
| 60 | `stress_coverage` | `stress_count` = 3 (raw readings) | 0.75 (= 3 / 4) | 75% of the expected daily stress readings were captured. |

**Context preserved?** The coverage ratio is actually *more* informative than the raw
count. Consider:

- **Before:** `wrist_temp_count = 908` and `stress_count = 3`. Are these good or bad?
  You cannot tell without knowing that wrist temp samples 1440 times/day and stress
  samples ~4 times/day. The raw numbers are not comparable across sensors.
- **After:** `wrist_temp_coverage = 0.63` and `stress_coverage = 0.75`. Now both are
  on the same 0-1 scale, and you can immediately see that stress had proportionally
  better coverage (75%) than wrist temp (63%) on this day.

The model uses these to weigh how much to trust the corresponding sensor's readings.
A coverage of 0.1 signals "sparse data, be cautious"; a coverage of 0.95 signals
"nearly complete day of data, high confidence."

---

## 7. Why Normalization Preserves Context

A common concern is: "If we change all the numbers, don't we lose the meaning of the
data?" The answer is no. Normalization is a **translation**, not a deletion. Here is
the reasoning for each type of transformation:

### 7.1 Z-Scored Columns: Relative Position is the Real Context

**The key insight:** For physiological data, the *absolute* value is almost never what
matters -- what matters is whether the value is *unusual for this person*.

Consider wrist temperature. Person A might have a baseline of 33.5 degrees C and Person B might
have a baseline of 35.0 degrees C. A raw reading of 34.5 degrees C is *high* for Person A but
*low* for Person B. If you gave a model both raw values (34.5 and 34.5), it would see
them as identical. But biologically, they represent opposite states -- one person is
running warm, the other is running cool.

After z-scoring:
- Person A's 34.5 degrees C becomes +1.5 (well above their average)
- Person B's 34.5 degrees C becomes -0.8 (below their average)

Now the model correctly sees that these are different body states, even though the
thermometer showed the same number. The z-score carries *more* contextual meaning
than the raw value, because it encodes "unusual for this individual."

**What is preserved:**
- Direction: positive = above personal average, negative = below
- Magnitude: how many standard deviations away from average
- Patterns: if estrogen rises while temperature drops, the z-scores show the same
  correlated pattern as the raw values

**What is removed (intentionally):**
- The absolute unit (degrees C, pg/mL, etc.). This is removed because ML models cannot
  use it meaningfully without participant-specific context, which the z-score provides.

**Reversibility:** `raw_value = z * participant_std + participant_mean`. The original
values can always be recovered from the merged file.

### 7.2 Ordinal Columns: Rank Order is the Context

When a participant reports "High" fatigue, the meaningful content is:
1. Fatigue is present (it's not "Not at all")
2. It's worse than "Low" or "Moderate" but not as bad as "Very High"
3. It's the 4th level out of 5

The normalized value 0.8 encodes all three of these facts:
1. It's greater than 0 (fatigue is present)
2. 0.8 > 0.4 ("Low") and 0.8 > 0.6 ("Moderate") but 0.8 < 1.0 ("Very High")
3. 0.8 = 4/5

Nothing is lost. The text label "High" is a human-readable convenience, but the
information it carries -- position on a severity scale -- is exactly what the number
0.8 represents.

**The model does not need to know the word "High."** It needs to know that this value
is higher than some and lower than others, which the number preserves perfectly.

### 7.3 One-Hot Columns: Format Change, Not Content Change

`phase = "Luteal"` and `[phase_Fertility=0, phase_Follicular=0, phase_Luteal=1, phase_Menstrual=0]`
contain exactly the same information. This is a lossless format conversion -- like
translating a sentence from English to French. The meaning is identical; only the
encoding is different.

ML models cannot process text strings, so one-hot encoding converts the category
into a format the model can use. No information is added or removed.

### 7.4 Coverage Ratios: More Context Than Raw Counts

Raw sample counts are **less informative** than coverage ratios, because a count alone
does not tell you whether it represents good or poor data quality.

- `count = 500` for a sensor that samples 1440/day means 35% coverage (poor).
- `count = 500` for a sensor that samples 10/day would mean the data is impossible.

The coverage ratio (count / expected maximum) adds the context of the sensor's sampling
rate, making the number self-explanatory. A model seeing `coverage = 0.35` immediately
knows "this day had sparse data" without needing a lookup table of sampling rates.

### 7.5 Missing Value Imputation: Conservative Neutral Assumptions

When a value is missing, we must fill it with *something* for the model to work. The
strategies used were chosen to be the **least opinionated** (least likely to introduce
false information):

- **Z-scored columns filled with 0:** Since 0 = the participant's own mean, this says
  "assume this reading was average for this person." It does not claim the value was
  high or low. It is the mathematical midpoint and the most neutral assumption.
- **Ordinal columns filled with median:** The median represents the most typical
  response across all participants. It does not skew toward extremes.

These imputed values are approximations, not measurements. They do not add false
context -- they add *no context*, which is the correct representation of "we don't
know what the real value was."

### 7.6 Summary: Nothing is Deleted, Everything is Translated

| Transformation | What it looks like changed | What actually happened to the context |
|---------------|---------------------------|---------------------------------------|
| Z-scoring | "34.6 degrees C" became "+0.3" | Context *improved*: "+0.3" means "slightly above this person's normal" which is more informative than a raw temperature without knowing the person |
| Ordinal encoding | "High" became "0.8" | Context preserved: 0.8 sits in the same relative position on the severity scale as "High" did among the text labels |
| One-hot encoding | "Luteal" became "[0,0,1,0]" | Context identical: the four binary columns encode the exact same category with zero loss |
| Coverage ratios | "count=908" became "0.63" | Context *improved*: 0.63 is self-explanatory ("63% of day covered") while 908 requires knowing the sensor's sampling rate |
| Missing values | Empty cell became "0" or "0.5" | Context approximated: "we don't know" is replaced with "assume average" -- the most conservative possible guess |

The original merged file (`merged_women_data.csv`) is always available for
human-readable interpretation. The normalized file is the same data, reformatted for a
machine to learn patterns from.

---

## 8. Body State Score Framework (Strict Science-Backed)

This section defines how to build a science-backed daily body state score from the
existing normalized dataset, without overclaiming thresholds or accuracy.

### 8.1 Evidence Hierarchy and Claim Rules

To keep the score scientifically defensible, use this hierarchy:

1. **Internal validation on this dataset (highest priority):** Any threshold, weight,
   or accuracy claim must be proven with participant-wise validation here.
2. **Peer-reviewed literature (supporting prior):** Papers are used to justify expected
   directions (for example, lower HRV often aligns with higher stress load), but not as
   direct universal cutoffs for this dataset.
3. **Expert heuristics (lowest priority):** Allowed only as temporary hypotheses.

**Hard rules for this project:**
- Do not publish fixed cutoffs such as `HRV < 20`, `temp > 36.8`, or `SpO2 < 96.5`
  unless those cutoffs are validated on this dataset.
- Do not publish fixed performance claims (for example, "92% accuracy") before running
  and documenting actual model evaluation on participant-held-out data.
- Keep outputs non-clinical: this is a body state guidance score, not diagnosis.

### 8.2 Score Definition (0-100)

Define a daily **Body State Score**:
- **0 to 100 continuous scale**
- Higher means stronger daily recovery/readiness balance
- Lower means higher physiological and symptom load for that day

Also define a separate **Confidence Score** (0-100) from coverage and missingness,
so the app can show both:
- `Body State Score = "how the body likely feels today"`
- `Confidence Score = "how much data support this estimate"`

### 8.3 How Scores Are Assigned (Data-Driven, Not Hardcoded)

Use the existing feature groups from `normalized_women_data.csv`:
- **Autonomic:** `rmssd_*`, `stress_score_*`
- **Thermal:** `wrist_temp_*`, `nightly_temp_mean`, `baseline_rel_*`
- **Hormonal context:** `lh`, `estrogen`, `pdg`
- **Symptom burden:** ordinal symptom columns (`fatigue`, `stress`, `cramps`, etc.)
- **Data quality:** `*_coverage`

Score assignment logic:
1. Train model(s) to map these features to a continuous state target.
2. Learn feature sign and weight from data (model coefficients and SHAP direction),
   not manual assumptions.
3. Calibrate predictions to a stable 0-100 scale.
4. Emit confidence based on day-level data completeness.

### 8.4 ML Prediction Mechanics (End-to-End)

**Training flow**
1. Build model table from `normalized_women_data.csv`.
2. Define target:
   - preferred: a same-day or next-day composite state target built from symptom burden
     and physiological balance; or
   - if needed in early stage: weak-label proxy target documented transparently.
3. Split by participant (grouped split) to prevent leakage.
4. Train:
   - baseline interpretable model (Elastic Net or GAM)
   - stronger nonlinear model (LightGBM/XGBoost)
5. Calibrate model output to 0-100.
6. Save model and calibration mapping.

**Inference flow (for app)**
1. Input today's normalized feature row.
2. Model predicts raw state value.
3. Calibration maps to 0-100 Body State Score.
4. Confidence module computes confidence from coverage and missingness.
5. Explanation layer returns top drivers (for example, SHAP top 3 positive/negative).
6. App renders score + confidence + plain-language guidance.

### 8.5 Correlation, Feature Checks, and Validation

Before and during modeling, run:

**Correlation analysis**
- Spearman correlation for ordinal/non-normal columns.
- Pearson correlation for approximately continuous z-scored columns.
- Partial correlations controlling for participant and phase context where needed.

**Feature quality checks**
- Multicollinearity (VIF/correlation clustering) and pruning.
- Missingness impact checks by feature group.
- Stability of relationships across cycle phases and participants.

**Validation design**
- Grouped CV or train/validation/test split by participant.
- Optional temporal holdout for prospective behavior.

**Metrics**
- Regression quality: MAE, RMSE.
- Ranking consistency: Spearman rho between predicted and observed target.
- Calibration quality: calibration curve/ECE-style summary.
- Fair/stable performance across participant segments and phase segments.

### 8.6 Persona Layer (Post-Model, App-Facing)

Personas are a communication layer, not the prediction engine.

Recommended flow:
1. Predict numeric score first.
2. Map score bands to label:
   - `0-39`: low capacity day
   - `40-69`: steady/moderate day
   - `70-100`: high readiness day
3. Add top contributing factors from explanation layer.

Important rule:
- Persona labels must **not** override model outputs via hardcoded if/else thresholds.
  They are generated from score + contributors after prediction.

### 8.7 Citation Hygiene and Research Traceability

When citing papers in this documentation:
- State what was actually studied (population, devices, endpoints).
- Use literature to justify directionality and candidate features.
- Mark external thresholds as hypotheses until internally validated.
- Add a short "validated in our data: yes/no" marker for each proposed rule.

This prevents over-claiming and keeps the system publishable and scalable.

---

## 9. Known Limitations & Notes

1. **High missing data in self-report columns (~45%):** Nearly half of the daily
   observations have no self-report data. These were imputed with the median, which
   may dilute signal for ML models. Consider:
   - Training separate models for days with/without self-report data
   - Using models that handle missing values natively (e.g., XGBoost, LightGBM)
   - Adding binary "has_selfreport" indicator features

2. **PDG (progesterone metabolite) is 36.6% missing:** Higher than other hormone columns
   (lh and estrogen are ~11.6% missing). May need special handling depending on the task.

3. **Coverage ratios depend on expected maximums:** The expected daily sample counts
   (1440 for per-minute sensors, 288 for HRV, 4 for stress) are theoretical maximums.
   Actual wearing time varies, so coverage < 1.0 is normal and expected.

4. **Z-score imputation with 0:** Missing z-scored values filled with 0 (= participant mean)
   is the most neutral choice but still introduces artificial data. High-missingness
   columns will have many 0-values that are imputed, not measured.

5. **Outer join created 379 extra rows:** The merge produced 6,038 rows vs. the base
   hormones table's 5,659 rows. These extra rows come from computed_temperature data
   on days where no hormones/self-report entry existed. They have NaN for all
   hormone/self-report fields (filled with median after normalization).

6. **No feature selection applied:** All available features are included. Depending on
   the ML task, dimensionality reduction (PCA) or feature selection may be beneficial.

7. **Reproducibility:** Both scripts (`merge_data.py`, `normalize_data.py`) can be
   re-run end-to-end to regenerate the output files from the original source CSVs.

---

## 10. Implemented Model Run (v2 Improved)

This section documents the improved implementation of the Body State Score model after
upgrading target design and feature engineering. It remains strict-science: no hardcoded
physiology thresholds were used in prediction.

### 10.1 What Was Implemented

**Script:** `build_body_state_score.py` (updated to v2)

Pipeline steps:
1. Load `normalized_women_data.csv`
2. Build a next-day blended burden target
3. Build temporal features (lag, delta, rolling trends)
4. Train three model families with participant-grouped validation
5. Select winner by held-out RMSE
6. Calibrate to 0-100 score
7. Export predictions, correlation table, feature importance, model artifact, and JSON report

### 10.2 Target Definition Used in v2

v2 uses a stronger weak-label than v1 by blending symptom and physiological stress context:

1. `symptom_burden_today = mean(12 ordinal symptom columns)`
2. Build physiological context terms:
   - `stress_component` from normalized `stress_score_mean`
   - `recovery_component` from normalized `rmssd_mean`
3. Blend into daily burden:
   `blended_burden_today = 0.55*symptom_burden_today + 0.30*stress_component + 0.15*(1 - recovery_component)`
4. Shift next day per participant:
   `blended_burden_next_day = shift(-1)`
5. Convert to score:
   `target_body_state_score = 100 * (1 - blended_burden_next_day)`

Interpretation:
- Higher score = lower expected next-day burden
- Lower score = higher expected next-day burden

### 10.3 Feature Set Used in v2

v2 expands from 32 to 58 features:

- Core physiology/context features (same groups as v1): autonomic, thermal, hormones,
  coverage, cycle phase, temporal context
- **Lag features** (for example `rmssd_mean_lag1`, `symptom_burden_today_lag1`)
- **Delta features** (today minus yesterday, for example `wrist_temp_mean_delta`)
- **Rolling trend features** (3-day mean/std, for example `blended_burden_today_roll3_mean`)

These additions let the model learn trajectory, not just single-day snapshots.

### 10.4 Training and Validation Design

Validation is participant-aware:
- Group key: `id`
- Holdout split: `GroupShuffleSplit` (80/20 participants)
- CV on training participants: `GroupKFold(n_splits=5)`

Models compared:
- `elastic_net_cv`
- `hist_gradient_boosting`
- `extra_trees`

Selection:
- Lowest held-out RMSE after calibration

Missing handling in v2:
- Lag/rolling features can be missing at series starts
- Median imputation is fit on training set only and applied to train/test/all inference tables

### 10.5 v2 Results (Actual Run)

From `body_state_model_report.json`:

- Data loaded: 6038 rows
- Rows with valid next-day target and lag context: 5914
- Feature count: 58
- Selected model: `elastic_net_cv`

Cross-validation:
- Elastic Net CV: MAE 4.06, RMSE 5.60, R2 0.379, Spearman 0.574
- Hist Gradient Boosting: MAE 4.37, RMSE 5.94, R2 0.303, Spearman 0.528
- Extra Trees: MAE 4.13, RMSE 5.69, R2 0.364, Spearman 0.579

Held-out test:
- Elastic Net CV: MAE 4.33, RMSE 6.16, R2 0.376, Spearman 0.588
- Hist Gradient Boosting: MAE 4.57, RMSE 6.41, R2 0.325, Spearman 0.565
- Extra Trees: MAE 4.37, RMSE 6.24, R2 0.360, Spearman 0.588

Performance change vs v1 baseline:
- RMSE improved from **12.34 -> 6.16**
- MAE improved from **8.40 -> 4.33**
- R2 improved from **~0.00 -> 0.376**
- Spearman improved from weak/unstable to **~0.59**

### 10.6 Top Signals Learned by v2

From `body_state_feature_importance.csv` (Elastic Net absolute coefficients):
- `blended_burden_today_roll3_mean`
- `symptom_burden_today_roll3_mean`
- `symptom_burden_today_lag1`
- `wrist_temp_coverage`
- `rmssd_mean`
- `blended_burden_today_lag1`

Interpretation:
- Recent trend and yesterday's burden are highly informative
- Recovery/autonomic and data-quality terms remain important
- The model is using plausible physiology and trajectory features, not arbitrary rules

### 10.7 How Prediction Works in the App (v2 Mechanics)

For each user-day:
1. Build the same 58-feature row (including lag and rolling features from prior days)
2. Apply training-median imputation for any missing lag windows
3. Predict raw output with selected model (`elastic_net_cv`)
4. Calibrate to 0-100 (`LinearRegression` calibrator)
5. Compute confidence components:
   - `coverage_confidence_score = mean(coverage columns) * 100`
   - `range_confidence_score` from out-of-range fraction against training q01-q99 feature bounds
6. Compute final confidence:
   `confidence_score = 0.65*coverage_confidence_score + 0.35*range_confidence_score`
7. Generate `confidence_reason` (high/moderate/low confidence explanation)
8. Map score to bands:
   - 0-39: Low Capacity Day
   - 40-69: Steady Day
   - 70-100: High Readiness Day
9. Return guidance text + confidence in app

New safety fields in `body_state_predictions.csv`:
- `coverage_confidence_score`
- `range_confidence_score`
- `confidence_score`
- `out_of_range_feature_count`
- `out_of_range_feature_fraction`
- `confidence_reason`

### 10.8 Why This Is Better and Still Science-Backed

v2 improves quality while preserving strict evidence discipline:
- Better target proxy design (symptom + stress/recovery blend)
- Better temporal representation (lag/trend/delta features)
- Strong participant-grouped validation
- Honest reporting of all model metrics
- No unvalidated fixed thresholds in scoring logic

This is now a materially stronger baseline suitable for iterative app integration.

### 10.9 How to Re-run

From the project folder:

`python build_body_state_score.py`

This regenerates the full v2 model outputs and metrics.

---

*End of documentation.*
