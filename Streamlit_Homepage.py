import streamlit as st
import base64
import pandas as pd
import joblib
import Connection as con

st.set_page_config(page_title="SpaceX Predictor", layout="centered")

def add_video_background(url):
    st.markdown(
        f"""
        <style>
        /* FULL SCREEN VIDEO BACKGROUND */
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
                allowfullscreen
            ></iframe>
        </div>
        """,
        unsafe_allow_html=True
    )

add_video_background(
    "https://www.youtube.com/embed/C3iHAgwIYtI?autoplay=1&mute=1&controls=0&loop=1&playlist=C3iHAgwIYtI"
)

model = joblib.load("launch_success_model.pkl")
cdf = con.get_data()
