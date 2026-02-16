# Menstrual Phase ML Model - Simple Explanation

This document explains your current model in simple words: what data it uses, how it learns, what files it generates, how good it is, and what to watch out for.

---

## 1) What this model is trying to do

Goal:
- Predict menstrual phase for a given day.

Label classes:
- `Follicular`
- `Fertility`
- `Luteal`
- `Menstrual`

This is **not** a clinical diagnosis model. It is a prediction helper for awareness/planning.

---

## 2) Which data is used

Main training file:
- `merged_women_data.csv`

Rows used for training:
- 5658 rows (rows where phase label exists)

Features include:
- Hormones: `lh`, `estrogen`, `pdg`
- Temperature features
- HRV/stress features
- Flow/symptom fields
- Time features (`day_in_study`, weekend)
- History/trend features created from previous days

---

## 3) What “history context” means

For each participant (`id`), data is sorted by day, then the script creates:

- `*_lag1`: yesterday’s value  
  Example: `lh_lag1`
- `*_delta`: today minus yesterday  
  Example: `lh_delta`
- `*_roll3_mean`, `*_roll3_std`: recent trend from prior days

Also created:
- `phase_lag1` = previous day phase

This helps the model understand day-to-day transitions, not just single-day snapshots.

---

## 4) How training is done

Script:
- `build_phase_prediction_model.py`

Pipeline:
1. Read data and keep rows with known phase label
2. Build temporal/history features
3. Split data by participant (`GroupShuffleSplit`) so same person does not leak between train/test
4. Train 3 model families with candidate tuning:
   - Logistic Regression
   - Random Forest
   - Extra Trees
5. Evaluate using grouped cross-validation (`GroupKFold`)
6. Pick winner by test macro F1
7. Save predictions, confusion matrix, feature relevance, model artifact, and report

---

## 5) Current best results

From latest `phase_model_report.json`:

- Feature strategy: `full_history`
- Selected model: `extra_trees`
- Test accuracy: **0.8704**
- Test balanced accuracy: **0.8610**
- Test macro F1: **0.8620**

All 3 models comparison (test):
- Extra Trees: 0.8704 acc
- Random Forest: 0.8672 acc
- Logistic Regression: 0.8381 acc

This is strong for a 4-class prediction task.

---

## 6) What output files mean

- `phase_predictions.csv`  
  Row-level true vs predicted phase on test set

- `phase_confusion_matrix.csv`  
  Where predictions are correct/wrong per class

- `phase_feature_correlation.csv`  
  Feature-to-label association (mutual information)

- `phase_model_comparison.csv`  
  Side-by-side metrics across the 3 models

- `phase_model_tuning_cv.csv`  
  Candidate tuning results per model

- `phase_prediction_model.joblib`  
  Saved model artifact for reuse

- `phase_model_report.json`  
  Complete metrics + metadata

---

## 7) Important caveat (very important)

Top association features include `phase_lag1_*` (previous day phase).

Why this matters:
- It can greatly increase accuracy because cycle phase usually changes gradually.
- This is valid **only if, in real use, previous-day phase is available** (from yesterday prediction or known data).

If previous-day phase is not available in production, accuracy will be lower than 0.87.

So for deployment, you should define one of two modes:
- **History-enabled mode** (higher accuracy)
- **Cold-start mode** (lower accuracy, no history assumptions)

---

## 8) Is this good for hackathon?

Yes.

You already have:
- Clear target
- Real preprocessing
- Multiple model comparison
- Leakage-safe grouped validation
- Strong metrics
- Model artifact and report files

Good demo line:
- “Non-clinical, data-driven phase prediction assistant for pre-informing likely cycle state.”

---

## 9) How to run

In PowerShell from this folder:

`python build_phase_prediction_model.py`

Optional feature strategy:
- Default: full-history high-accuracy mode
- Hormone-primary mode:

`$env:PHASE_FEATURE_MODE="hormone_primary"`
`python build_phase_prediction_model.py`

---

## 10) In one sentence

Your model learns phase patterns from hormones + physiology + daily trends, compares 3 model families safely, and currently predicts phase with about **87% accuracy** in grouped participant testing.
