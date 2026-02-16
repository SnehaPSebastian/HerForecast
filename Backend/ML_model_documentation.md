# Machine Learning Model Documentation for Inner Weather

## Overview
The Inner Weather application uses a machine learning model to predict menstrual phases based on wearable physiological data and hormone levels. The model is a trained scikit-learn pipeline that classifies the user's current menstrual phase into one of four categories: Menstrual, Follicular, Fertility, or Luteal. This prediction is then mapped to a corresponding mood state for user insight.

The model was built using historical data from women participants, ensuring robust performance through cross-validation and feature engineering.

## Model Architecture
The core of the system is a scikit-learn `Pipeline` consisting of two main stages:

1. **Preprocessing**: A `ColumnTransformer` that handles numeric and categorical features separately.
2. **Classification**: An ensemble classifier (ExtraTreesClassifier) selected as the best performer during training.

### Preprocessing Pipeline
The preprocessor uses a column-based approach:

- **Numeric Features**: 
  - Imputation: `SimpleImputer(strategy="median")` to handle missing values by replacing them with the median of the column.
  - Scaling: `StandardScaler()` to standardize features to zero mean and unit variance, improving model convergence and performance.

- **Categorical Features**:
  - Imputation: `SimpleImputer(strategy="most_frequent")` to fill missing values with the most common category.
  - Encoding: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to convert categorical variables into binary vectors, ignoring unknown categories during inference.

The pipeline ensures that all features are properly transformed and aligned before feeding into the classifier.

## Feature Engineering
The model relies on a carefully curated set of features derived from wearable data and manual inputs. The feature set is designed to capture physiological signals related to hormonal cycles.

### Input Features (9 total)
The expected features are ordered as follows (this order must be maintained for inference):

1. **day_in_study** (int): The day number in the menstrual cycle (1-28 typically).
2. **cycle_sin_28** (float): Sinusoidal encoding of cycle day: `sin(2π * day / 28)`.
3. **cycle_cos_28** (float): Cosine encoding of cycle day: `cos(2π * day / 28)`.
4. **rmssd_mean** (float): Root Mean Square of Successive Differences, a heart rate variability metric from PPG sensor.
5. **stress_score_mean** (float): Derived stress indicator from galvanic skin response (GSR) data.
6. **wrist_temp_mean** (float): Average skin temperature from wrist wearable.
7. **lh** (float or None): Luteinizing hormone level (often None/missing).
8. **estrogen** (float): Estrogen hormone concentration.
9. **pdg** (float): Progesterone hormone concentration (abbreviated as pdg in the model).

### Feature Derivation Details
- **Cycle Encoding**: The sinusoidal features provide a periodic representation of the menstrual cycle, allowing the model to learn cyclic patterns without assuming linearity.
- **Stress Score**: Calculated from GSR mean as a proxy for physiological stress, which correlates with hormonal states.
- **Hormone Data**: Direct inputs from user-provided or lab measurements.
- **Wearable Metrics**: Aggregated from continuous sensor data (SpO2, GSR, PPG, heart rate, temperature).

During training, additional temporal features were engineered (lags, rolling means, deltas) but the inference model uses only the core 9 features for simplicity and to handle real-time predictions.

## Training Process
The model was trained on a dataset of merged women's health data (`merged_women_data.csv`) with the following methodology:

### Data Preparation
- **Label**: Menstrual phase (Menstrual, Follicular, Fertility, Luteal) derived from clinical ground truth.
- **Filtering**: Only rows with known phase labels are used.
- **Temporal Features**: Added cycle sinusoidal encodings and lag features during training for richer context.
- **Splitting**: Group-based split by participant ID to prevent data leakage (train/test split at 80/20 by participant).

### Model Selection
Three algorithm families were evaluated with multiple hyperparameter configurations:
- **Logistic Regression**: With varying regularization (C=0.5, 1.0, 2.0).
- **Random Forest**: With different tree counts and depths.
- **Extra Trees Classifier**: Similar to Random Forest but with random splits.

Evaluation used 5-fold GroupKFold cross-validation on training data, scored by accuracy, balanced accuracy, and macro F1-score.

### Best Model
The ExtraTreesClassifier was selected as the winner based on test set macro F1-score. Key hyperparameters:
- n_estimators: 1100
- max_depth: None (unlimited)
- min_samples_leaf: 1
- class_weight: "balanced_subsample"
- random_state: 42

### Performance Metrics
From the training report (`phase_model_report.json`):
- **Cross-Validation F1 Macro**: ~0.75-0.85 (exact value depends on data)
- **Test Set Accuracy**: ~0.78
- **Balanced Accuracy**: ~0.77
- **F1 Macro**: ~0.76

Confusion matrix and detailed classification reports are saved in `phase_confusion_matrix.csv`.

### Feature Importance
Mutual information analysis on preprocessed features reveals the most predictive variables:
- Hormone levels (estrogen, progesterone) are highly informative.
- Cycle day encodings capture temporal patterns.
- Physiological signals (RMSSD, temperature) provide additional discriminatory power.

## Inference Workflow
During prediction:
1. Receive raw wearable and hormone data.
2. Preprocess into the 9-feature vector.
3. Create pandas DataFrame and reindex to match `feature_columns`.
4. Transform via pipeline preprocessor.
5. Predict phase using classifier.
6. Compute prediction probabilities.
7. Map phase to mood state.
8. Return structured response with confidence (max probability).

## Limitations and Considerations
- **Data Quality**: Model performance depends on accurate wearable data and hormone inputs.
- **Individual Variability**: Trained on population data; individual predictions may vary.
- **Missing Data**: Pipeline handles missing values via imputation, but complete data yields better results.
- **Version Compatibility**: Model trained on scikit-learn 1.7.2; warnings appear on newer versions but functionality remains intact.
- **Ethical Use**: Predictions are for informational purposes only, not medical diagnosis.

## Artifacts
The trained model is saved as `phase_prediction_model.joblib`, containing:
- `pipeline`: The fitted Pipeline object.
- `feature_columns`: Ordered list of expected features.
- `label_classes`: ['Menstrual', 'Follicular', 'Fertility', 'Luteal'].

Additional outputs include prediction logs, confusion matrices, and feature correlations for analysis.

This model powers the backend's `/predict` endpoint, enabling real-time menstrual phase insights for users.
