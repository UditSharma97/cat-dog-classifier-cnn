import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from src.model_loader import load_model
from src.predict import predict_image

# --- Page Configuration ---
st.set_page_config(page_title="Cat vs Dog Image Classification Using CNN", page_icon="🐾", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    /* Main app pure black background */
    .stApp {
        background-color: #000000;
    }
    
    /* MAKE THE TOP NAVBAR PURE BLACK */
    [data-testid="stHeader"] { 
        background-color: #000000 !important; 
    }
    
    /* Padding adjustments */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 120px !important;
    }
    
    /* Global Typography */
    html, body, [class*="css"], p, h2, h3, h4, h5, h6, span, label, li {
        color: #FFFFFF !important;
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #4da6ff !important; 
    }

    /* --- THE VERTICAL PARTITION LINE --- */
    /* Target ONLY the first column of the main horizontal block */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
        border-right: 1px solid #ffffff;
        padding-right: 2rem;
    }
    /* Prevent the line from appearing on nested columns (like sample images) */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
        border-right: none;
        padding-right: 0;
    }

    /* Fixed Footer CSS */
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #000000;
        padding: 15px 0;
        text-align: center;
        border-top: 1px solid #222222;
        z-index: 100;
    }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# Loading model
try:
    model = load_model()
except Exception:
    model = None

# --- MAIN PAGE HEADER (Custom Font Weight) ---
st.markdown("<h1 style='font-weight: 500; color: #FFFFFF; font-family: \"Inter\", sans-serif; margin-bottom: 0;'>🐾 Cat vs Dog Image Classification Using CNN</h1>", unsafe_allow_html=True)
st.divider()

# --- LAYOUT ARCHITECTURE ---
col_info, col_app = st.columns([1, 2.5], gap="large")

# --- LEFT COLUMN (Sample Data) ---
with col_info:
    st.markdown("### 🖼️ Sample Data")
    st.markdown("<p style='font-size: 13px; color: #888888;'>Right-click and save to test the model.</p>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        try:
            st.image("sample_data/cat.jpg", caption="Cat", use_container_width=True)
            st.image("sample_data/cat.86.jpg", caption="Cat", use_container_width=True)
        except FileNotFoundError:
            pass 
    with col_s2:
        try:
            st.image("sample_data/images (1).jpg", caption="Dog", use_container_width=True)
            st.image("sample_data/images.jpg", caption="Dog", use_container_width=True)

        except FileNotFoundError:
            pass
            
    st.markdown("<br><br>", unsafe_allow_html=True)

# --- RIGHT COLUMN (Application & Results) ---
with col_app:
    st.markdown("### 🔍 Upload & Classify")
    uploaded_file = st.file_uploader("Drop your image here...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        inner_col1, inner_col2 = st.columns([1, 1], gap="medium")
        
        with inner_col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
        with inner_col2:
            st.subheader("Analysis Results")
            
            if st.button("Run Classification 🚀", type="primary", use_container_width=True):
                if model is not None:
                    with st.spinner("Analyzing image features..."):
                        prob_cat, prob_dog = predict_image(model, image)
                        
                        if prob_dog > 0.5:
                            winner_label = "Dog"
                            confidence = prob_dog
                        else:
                            winner_label = "Cat"
                            confidence = prob_cat
                            
                        st.success(f"Prediction: **{winner_label}**")
                        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                        
                        st.divider()
                        st.markdown("**Probability Breakdown**")
                        st.write(f"Dog: {prob_dog * 100:.1f}%")
                        st.progress(prob_dog)
                        
                        st.write(f"Cat: {prob_cat * 100:.1f}%")
                        st.progress(prob_cat)
                else:
                    st.error("Model failed to load.")

# --- FULL-WIDTH FOOTER ---
st.markdown(
    """
    <div class="fixed-footer">
        <p style="margin-bottom: 5px; color: #888888; font-size: 14px;">Developed by <b>Udit Sharma</b></p>
        <p style="font-size: 13px; margin-bottom: 5px;">
            <a href="https://github.com/UditSharma97" style="color: #4da6ff; text-decoration: none;">GitHub</a> • 
            <a href="https://www.linkedin.com/in/udit-sharma-029879294/" style="color: #4da6ff; text-decoration: none;">LinkedIn</a>
        </p>
        <p style="font-size: 12px; color: #555555; margin-bottom: 0;">
            © 2026 Cat vs Dog Classifier. Powered by Streamlit & TensorFlow.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)