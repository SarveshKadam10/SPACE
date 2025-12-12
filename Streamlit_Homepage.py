import streamlit as st
import pandas as pd
import joblib
import Connection as con


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="SpaceX Launch Success Predictor", layout="wide")


# ---------------------------------------------------------
# BACKGROUND VIDEO
# ---------------------------------------------------------
def add_video_background(url: str):
    st.markdown(
        f"""
        <style>
        .video-bg-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
        }}
        .video-bg-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            border: none;
            pointer-events: none;
            object-fit: cover;
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>

        <div class="video-bg-container">
            <iframe
                src="{url}"
                frameborder="0"
                allow="autoplay; fullscreen"
                allowfullscreen></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )


add_video_background(
    "https://www.youtube.com/embed/C3iHAgwIYtI?autoplay=1&mute=1&controls=0&loop=1&playlist=C3iHAgwIYtI"
)


# ---------------------------------------------------------
# LOAD MODEL + DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("launch_success_model.pkl")


@st.cache_data
def load_data():
    return con.get_data()


model = load_model()
cdf = load_data()

cdf = cdf.dropna(subset=["success"])


# ---------------------------------------------------------
# FEATURE LISTS (same as Model.py)
# ---------------------------------------------------------
numeric_features = [
    "payload_mass_kg",
    "core_reuse_count",
    "core_block",
    "core_rtls_attempts",
    "core_rtls_landings",
    "core_asds_attempts",
    "core_asds_landings",
    "capsule_reuse_count",
    "water_landings",
    "land_landings",
]

categorical_features = [
    "rocket_id",
    "launchpad_status",
    "launchpad_region",
    "core_status",
    "capsule_status",
    "payload_type",
    "payload_orbit",
    "payload_regime",
    "payload_ref_sys",
    "ship_type",
]

binary_features = [
    "fairings_reused",
    "fairings_recovery_attempts",
    "fairings_recovered",
    "ship_active",
]

feature_cols = numeric_features + categorical_features + binary_features


# ---------------------------------------------------------
# SIDEBAR MODE SELECTOR
# ---------------------------------------------------------
mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Predict Historical Launch", "Predict NEW Launch"]
)


# ---------------------------------------------------------
# MODE 1 — HISTORICAL PREDICTION
# ---------------------------------------------------------
if mode == "Predict Historical Launch":
    st.title("🚀 Historical Launch Prediction")

    launch_ids = cdf["launch_id"].astype(str).tolist()
    selected_id = st.selectbox("Select a launch ID:", launch_ids)

    row = cdf[cdf["launch_id"].astype(str) == selected_id].iloc[0]

    st.subheader("Launch details")
    st.json(
        {
            "launch_name": row.get("launch_name"),
            "date": row.get("date"),
            "rocket_id": row.get("rocket_id"),
            "launchpad_id": row.get("launchpad_id"),
        }
    )

    # CLEANING (same logic as training)
    for col in numeric_features:
        row[col] = 0 if pd.isna(row[col]) else row[col]

    for col in binary_features:
        row[col] = 0 if pd.isna(row[col]) else row[col]

    for col in categorical_features:
        val = row[col]
        if pd.isna(val):
            val = "Unknown"
        row[col] = str(val)

    X_single = pd.DataFrame([row[feature_cols].to_dict()])

    if st.button("Predict Success"):
        pred = model.predict(X_single)[0]
        prob = model.predict_proba(X_single)[0][1]

        st.subheader("Model Prediction")
        if pred == 1:
            st.success(f"Predicted: SUCCESS (Probability: {prob:.2%})")
        else:
            st.error(f"Predicted: FAILURE (Probability: {prob:.2%})")

        st.subheader("Actual Outcome")
        actual = bool(row["success"])
        st.write("Actual: **SUCCESS**" if actual else "Actual: **FAILURE**")


# ---------------------------------------------------------
# MODE 2 — NEW LAUNCH PREDICTION
# ---------------------------------------------------------
else:
    st.title("🚀 Predict a NEW Launch")

    st.markdown("Manually enter mission parameters below:")

    col1, col2, col3 = st.columns(3)

    # ------------------ NUMERIC FEATURES ------------------
    with col1:
        payload_mass_kg = st.number_input("Payload Mass (kg)", min_value=0, value=3000)
        core_reuse_count = st.number_input("Core Reuse Count", min_value=0, value=1)
        core_rtls_attempts = st.number_input("RTLS Attempts", min_value=0, value=0)

    with col2:
        core_block = st.number_input("Core Block Version", min_value=0, value=5)
        core_rtls_landings = st.number_input("RTLS Landings", min_value=0, value=0)
        core_asds_attempts = st.number_input("ASDS Attempts", min_value=0, value=1)

    with col3:
        core_asds_landings = st.number_input("ASDS Landings", min_value=0, value=1)
        capsule_reuse_count = st.number_input("Capsule Reuse Count", min_value=0, value=0)
        water_landings = st.number_input("Water Landings", min_value=0, value=0)

    land_landings = st.number_input("Land Landings", min_value=0, value=0)

    # ------------------ BINARY FEATURES ------------------
    st.subheader("Recovery & Fairings")

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        fairings_reused = st.selectbox("Fairings Reused?", [0, 1])

    with col5:
        fairings_recovery_attempts = st.selectbox("Fairing Recovery Attempt?", [0, 1])

    with col6:
        fairings_recovered = st.selectbox("Fairings Recovered?", [0, 1])

    with col7:
        ship_active = st.selectbox("Ship Active?", [0, 1])

    # ------------------ CATEGORICAL FEATURES ------------------
    st.subheader("Rocket / Payload / Launchpad")

    col8, col9, col10 = st.columns(3)

    with col8:
        rocket_id = st.selectbox("Rocket ID", sorted(cdf["rocket_id"].unique()))
        core_status = st.selectbox("Core Status", sorted(cdf["core_status"].unique()))
        payload_type = st.selectbox("Payload Type", sorted(cdf["payload_type"].unique()))

    with col9:
        launchpad_status = st.selectbox("Launchpad Status", sorted(cdf["launchpad_status"].unique()))
        capsule_status = st.selectbox("Capsule Status", sorted(cdf["capsule_status"].unique()))
        payload_orbit = st.selectbox("Payload Orbit", sorted(cdf["payload_orbit"].unique()))

    with col10:
        launchpad_region = st.selectbox("Launchpad Region", sorted(cdf["launchpad_region"].unique()))
        payload_regime = st.selectbox("Payload Regime", sorted(cdf["payload_regime"].unique()))
        payload_ref_sys = st.selectbox("Payload Reference System", sorted(cdf["payload_ref_sys"].unique()))
        ship_type = st.selectbox("Ship Type", sorted(cdf["ship_type"].unique()))

    # ---------------------------------------------------------
    # BUILD A NEW LAUNCH FEATURE ROW
    # ---------------------------------------------------------
    new_launch = {
        "payload_mass_kg": payload_mass_kg,
        "core_reuse_count": core_reuse_count,
        "core_block": core_block,
        "core_rtls_attempts": core_rtls_attempts,
        "core_rtls_landings": core_rtls_landings,
        "core_asds_attempts": core_asds_attempts,
        "core_asds_landings": core_asds_landings,
        "capsule_reuse_count": capsule_reuse_count,
        "water_landings": water_landings,
        "land_landings": land_landings,
        "fairings_reused": fairings_reused,
        "fairings_recovery_attempts": fairings_recovery_attempts,
        "fairings_recovered": fairings_recovered,
        "ship_active": ship_active,
        "rocket_id": str(rocket_id),
        "launchpad_status": str(launchpad_status),
        "launchpad_region": str(launchpad_region),
        "core_status": str(core_status),
        "capsule_status": str(capsule_status),
        "payload_type": str(payload_type),
        "payload_orbit": str(payload_orbit),
        "payload_regime": str(payload_regime),
        "payload_ref_sys": str(payload_ref_sys),
        "ship_type": str(ship_type),
    }

    X_new = pd.DataFrame([new_launch])

    # ---------------------------------------------------------
    # PREDICT NEW LAUNCH
    # ---------------------------------------------------------
    if st.button("Predict NEW Launch Success"):
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]

        st.subheader("Model Prediction for NEW Launch")
        if pred == 1:
            st.success(f"SUCCESS LIKELY — Probability: {prob:.2%}")
        else:
            st.error(f"FAILURE LIKELY — Probability: {prob:.2%}")
