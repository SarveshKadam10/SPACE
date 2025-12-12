from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import Connection as con

# Load merged data
cdf = con.get_data()

# ================================
# IMPORTANT FEATURES ONLY
# ================================
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

# Only keep needed columns
cdf = cdf[all_features + ["success"]]

# Clean missing data
cdf[numeric_features] = cdf[numeric_features].fillna(0)
cdf[categorical_features] = cdf[categorical_features].fillna("Unknown")

X = cdf[all_features]
y = cdf["success"].astype(int)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

# Logistic Regression Model
model = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))
joblib.dump(model, "launch_success_model.pkl")
print("Model saved as launch_success_model.pkl")
