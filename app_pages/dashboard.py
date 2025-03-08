import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_sample_data
from src.model import load_trained_model
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Loads and returns the trained model from the absolute path using caching to avoid reloading each time.
    """
    model_path = os.path.abspath("src/model.h5")
    if not os.path.exists(model_path):
        st.error(f"Error: Model not found at {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def predict_image(model, image):
    """
    Receives the model and a preprocessed image, and performs the prediction.
    """
    image = image.reshape((1, *image.shape))
    prediction = model.predict(image)
    return prediction

def get_sample_data():
    """
    Returns a DataFrame with sample data for visualizations.
    """
    data = pd.DataFrame({
        'Month': range(1, 13),
        'Value': np.random.randint(50, 150, 12)
    })
    return data.set_index('Month')

def list_images_in_folder(folder, extensions=('.jpg', '.jpeg', '.png')):
    """
    Lists the image files in a folder.
    """
    if os.path.exists(folder):
        return [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    return []

# Configure Streamlit page layout
st.set_page_config(page_title="Mildew Detection Dashboard", layout="wide")

# CSS style
st.markdown("""
<style>
    /* Default style for light mode */
    .sub-title { font-size: 26px; color: #2c3e50; margin-bottom: 20px; text-align: center; }
    .section-title { font-size: 22px; color: #34495e; margin-top: 20px; }
    .content { font-size: 18px; color: #2c3e50; line-height: 1.5; }
    .legend { font-size: 16px; color: #7f8c8d; text-align: center; }
    
    /* Styles for dark mode */
    @media (prefers-color-scheme: dark) {
      .sub-title { color: #f0f0f0 !important; }
      .section-title { color: #e0e0e0 !important; }
      .content { color: #f0f0f0 !important; }
      .legend { color: #c0c0c0 !important; }
    }
</style>
""", unsafe_allow_html=True)