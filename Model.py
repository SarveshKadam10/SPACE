from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import Connection as con
import pandas as pd

# -----------------------------
# LOAD FINAL MERGED DATA
# -----------------------------
cdf = con.get_data()

numeric_features = [
    "payload_mass_kg","core_reuse_count","core_block",
    "core_rtls_attempts","core_rtls_landings",
    "core_asds_attempts","core_asds_landings",
    "capsule_reuse_count","water_landings","land_landings"
]

categorical_features = [
    "rocket_id","launchpad_status","launchpad_region",
    "core_status","capsule_status","payload_type",
    "payload_orbit","payload_regime","payload_ref_sys",
    "ship_type"
]

binary_features = [
    "fairings_reused","fairings_recovery_attempts",
    "fairings_recovered","ship_active"
]

# -----------------------------
# FIX MISSING VALUES
# -----------------------------
cdf[numeric_features] = cdf[numeric_features].fillna(0)
cdf[binary_features] = cdf[binary_features].fillna(0)
cdf[categorical_features] = cdf[categorical_features].fillna("Unknown")

# -----------------------------
# TRAINING INPUTS
# -----------------------------
X = cdf[numeric_features + categorical_features + binary_features]
y = cdf["success"]

# -----------------------------
# PIPELINE
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
])

# -----------------------------
# TRAIN
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "launch_success_model.pkl")
print("Model saved.")
