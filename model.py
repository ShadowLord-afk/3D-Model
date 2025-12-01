import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lazypredict.Supervised import LazyClassifier

# Load the dataset
df = pd.read_csv("dataset/synthetic_medical_symptoms_dataset.csv")
df.head()

# Initial data exploration
print(df["diagnosis"].unique())
print(df.info())
print(df["loss_smell"].value_counts(), df["loss_taste"].value_counts())

# Data cleaning and preprocessing
df = df.dropna()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix - Before Feature Engineering")
plt.show()

# Feature engineering
# Create age groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 65, 100],
    labels=["pediatric", "adolescent", "adult", "elderly"],
)
df["age_group"] = df["age_group"].astype(str)
# Create symptom burden feature
symptom_cols = [
    "fever",
    "cough",
    "fatigue",
    "headache",
    "muscle_pain",
    "nausea",
    "loss_smell",
    "loss_taste",
    "vomiting",
    "diarrhea",
]
df["symptom_burden"] = df[symptom_cols].sum(axis=1)

# Lab Value Ratios
df["wbc_to_hemoglobin"] = df["wbc_count"] / (df["hemoglobin"] + 0.1)
df["crp_to_glucose"] = df["crp_level"] / (df["glucose_level"] + 0.1)

# Vital Sign Deviations
df["temp_deviation"] = abs(df["temperature_c"] - 98.6)
df["heart_rate_abnormal"] = ((df["heart_rate"] < 60) | (df["heart_rate"] > 100)).astype(
    int
)
df["oxygen_low"] = (df["oxygen_saturation"] < 95).astype(int)
df["glucose_abnormal"] = (
    (df["glucose_level"] < 70) | (df["glucose_level"] > 140)
).astype(int)

# Additional features
df["fever_indicator"] = (df["temperature_c"] > 37.5).astype(int)
df["blood_pressure_high"] = (
    (df["systolic_bp"] > 140) | (df["diastolic_bp"] > 90)
).astype(int)
df["wbc_abnormal"] = ((df["wbc_count"] < 4.5) | (df["wbc_count"] > 11.0)).astype(int)

# Looking at Data Correlations Again
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix - After Feature Engineering")
plt.show()

# Model Building and Evaluation
X = df.drop("diagnosis", axis=1).copy()
y = df["diagnosis"].copy()

# Print data types BEFORE encoding
print("\nData types BEFORE encoding:")
print(X.dtypes)

# ✅ ENCODE CATEGORICAL VARIABLES - Use .astype() instead of loop
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}")

# ✅ ENCODE TARGET VARIABLE
le_target = LabelEncoder()
y = le_target.fit_transform(y.astype(str))

# Print data types AFTER encoding
print("\nData types AFTER encoding:")
print(X.dtypes)
print(f"\nX shape: {X.shape}")
print(f"All numeric: {X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()}")

# ✅ SPLIT AFTER ENCODING (all data is numeric now)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (CRITICAL for SGDClassifier)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Using LazyPredict to compare models
print("\n" + "=" * 80)
print("EVALUATING ALL MODELS WITH LAZYPREDICT...")
print("=" * 80)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)

# Extract and display results
results_df = models.copy()
results_df = results_df.sort_values("Accuracy", ascending=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON RESULTS (Sorted by Accuracy)")
print("=" * 80)
print(results_df)

# HYPERPARAMETER TUNING FOR SGDC
print(f"\n{'=' * 80}")
print("HYPERPARAMETER TUNING FOR SGDC...")
print(f"{'=' * 80}")

sgdc_param_grid = {
    "loss": ["log_loss", "squared_hinge"],
    "penalty": ["l2", "l1"],
    "alpha": [0.001, 0.01],
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.01, 0.1],
    "max_iter": [1000, 2000],
}

sgdc = SGDClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    sgdc, sgdc_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# Evaluate tuned model
tuned_sgdc = grid_search.best_estimator_
y_pred_tuned = tuned_sgdc.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

# Baseline SGDC accuracy
baseline_sgdc = SGDClassifier(random_state=42)
baseline_sgdc.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_sgdc.predict(X_test_scaled)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"\n{'=' * 80}")
print("SGDC ACCURACY COMPARISON")
print(f"{'=' * 80}")
print(f"Baseline SGDC Test Accuracy: {baseline_accuracy:.4f}")
print(f"Tuned SGDC Test Accuracy: {tuned_accuracy:.4f}")
print(f"Improvement: {(tuned_accuracy - baseline_accuracy) * 100:.2f}%")

print("\nTuned SGDC Classification Report:")
print(classification_report(y_test, y_pred_tuned))

# Confusion Matrix for tuned SGDC
plt.figure(figsize=(10, 8))
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Blues")
plt.title("Tuned SGDC - Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(
    tuned_sgdc, X_train_scaled, y_train, cv=5, scoring="accuracy"
)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
