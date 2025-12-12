import streamlit as st
import pandas as pd
import joblib
import Connection as con

# ---------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------
st.set_page_config(page_title="SpaceX Success Predictor", layout="wide")


# ---------------------------------------------------------
# Background video
# ---------------------------------------------------------
def add_video_background(url: str):
    st.markdown(
        f"""
        <style>
        .video-bg-container {{
            position: fixed; top:0; left:0;
            width:100vw; height:100vh;
            overflow:hidden; z-index:-1;
        }}
        .video-bg-container iframe {{
            position:absolute; width:100vw; height:100vh;
            border:none; object-fit:cover;
            pointer-events:none;
        }}
        .stApp {{ background: transparent !important; }}
        </style>

        <div class="video-bg-container">
            <iframe src="{url}" autoplay loop mute></iframe>
        </div>
        """,
        unsafe_allow_html=True
    )


add_video_background(
    "https://www.youtube.com/embed/C3iHAgwIYtI?autoplay=1&mute=1&controls=0&loop=1"
)


# ---------------------------------------------------------
# Load model + data
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("launch_success_model.pkl")


@st.cache_data
def load_df():
    return con.get_data()


model = load_model()
cdf = load_df()

cdf = cdf.dropna(subset=["success"])  # ensure clean rows


# ---------------------------------------------------------
# Feature lists (IMPORTANT FEATURES ONLY)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Sidebar: prediction modes
# ---------------------------------------------------------
mode = st.sidebar.radio("Prediction Mode", ["Historical Launch", "New Launch"])


# ---------------------------------------------------------
# MODE 1 — Historical Prediction
# ---------------------------------------------------------
if mode == "Historical Launch":
    st.title("🚀 Historical Launch Prediction")

    launch_list = cdf["launch_id"].astype(str).tolist()
    selected = st.selectbox("Choose a launch ID", launch_list)

    row = cdf[cdf["launch_id"].astype(str) == selected].iloc[0]

    st.subheader("Launch Overview")
    st.json({
        "launch_name": row["launch_name"],
        "date": row["date"],
        "rocket_id": row["rocket_id"],
        "launchpad_status": row["launchpad_status"],
        "payload_orbit": row["payload_orbit"],
    })

    # Clean + prepare row
    clean_row = {}

    # numeric
    for col in numeric_features:
        v = row[col]
        clean_row[col] = 0 if pd.isna(v) else float(v)

    # categorical
    for col in categorical_features:
        v = row[col]
        clean_row[col] = "Unknown" if pd.isna(v) else str(v)

    X = pd.DataFrame([clean_row])

    if st.button("Predict Success"):
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        st.subheader("Prediction")
        if pred == 1:
            st.success(f"Success Likely — {prob:.2%}")
        else:
            st.error(f"Failure Likely — {prob:.2%}")

        st.subheader("Actual Outcome")
        st.write("SUCCESS" if row["success"] else "FAILURE")


# ---------------------------------------------------------
# MODE 2 — NEW LAUNCH PREDICTION
# ---------------------------------------------------------
else:
    st.title("🚀 Predict a NEW SpaceX Launch")

    st.markdown("Enter your mission parameters below:")

    # -------- numeric inputs --------
    col1, col2, col3 = st.columns(3)

    with col1:
        payload_mass_kg = st.number_input("Payload Mass (kg)", 0, 60000, 3000)
        core_reuse_count = st.number_input("Core Reuse Count", 0, 20, 1)

    with col2:
        core_block = st.number_input("Core Block Version", 1, 10, 5)
        core_rtls_attempts = st.number_input("RTLS Attempts", 0, 20, 1)

    with col3:
        core_rtls_landings = st.number_input("RTLS Landings", 0, 20, 1)
        core_asds_attempts = st.number_input("ASDS Attempts", 0, 20, 1)

    core_asds_landings = st.number_input("ASDS Landings", 0, 20, 1)

    # -------- categorical inputs --------
    st.subheader("Rocket / Pad / Payload Parameters")

    colA, colB, colC = st.columns(3)

    with colA:
        rocket_id = st.selectbox(
            "Rocket ID", sorted(cdf["rocket_id"].dropna().astype(str).unique())
        )

    with colB:
        launchpad_status = st.selectbox(
            "Launchpad Status", sorted(cdf["launchpad_status"].unique())
        )

    with colC:
        payload_orbit = st.selectbox(
            "Payload Orbit", sorted(cdf["payload_orbit"].unique())
        )

    # Build row for prediction
    new_row = {
        "payload_mass_kg": payload_mass_kg,
        "core_reuse_count": core_reuse_count,
        "core_block": core_block,
        "core_rtls_attempts": core_rtls_attempts,
        "core_rtls_landings": core_rtls_landings,
        "core_asds_attempts": core_asds_attempts,
        "core_asds_landings": core_asds_landings,
        "rocket_id": str(rocket_id),
        "launchpad_status": str(launchpad_status),
        "payload_orbit": str(payload_orbit),
    }

    X_new = pd.DataFrame([new_row])

    if st.button("Predict NEW Launch Success"):
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]

        st.subheader("Prediction for NEW Launch")
        if pred == 1:
            st.success(f"SUCCESS likely — {prob:.2%}")
        else:
            st.error(f"FAILURE likely — {prob:.2%}")
