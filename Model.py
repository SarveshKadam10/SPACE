import pandas as pd
import joblib
import Connection as con
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data
cdf = con.get_data()

# IMPORTANT FEATURES ONLY
numeric_features = [
    "payload_mass_kg",
    "core_reuse_count",
    "core_block",
    "core_rtls_attempts",
    "core_rtls_landings",
    "core_asds_attempts",
    "core_asds_landings",
]

categorical_features = [
    "rocket_id",
    "launchpad_status",
    "payload_orbit",
]

all_features = numeric_features + categorical_features

df = cdf[all_features + ["success"]]

# Clean numeric and categorical missing data
df[numeric_features] = df[numeric_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("Unknown")

X = df[all_features]
y = df["success"].astype(int)

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

# Random Forest Model
model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    ))
])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "launch_success_model.pkl")
print("Model saved as launch_success_model.pkl")
