import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from src.config import IMG_SIZE, MODEL_PATH


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(pil_img):
    """Convert PIL image to normalized model input array"""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img / 255.0, axis=0), img


def get_mask(pred):
    """Threshold model output to binary mask"""
    return (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255


# ── UI ─────────────────────────────────────────────────────────

st.title("Human Segmentation System")
st.write("Upload an image to generate a segmentation mask.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model()
    pil_img = Image.open(uploaded_file)

    img_input, img_resized = preprocess(pil_img)

    with st.spinner("Running model..."):
        pred = model.predict(img_input)

    mask = get_mask(pred)

    # Red overlay on segmented human region
    overlay = img_resized.copy()
    overlay[mask == 255] = [255, 0, 0]
    blended = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Predicted Mask")
        st.image(mask, use_container_width=True)
    with col3:
        st.subheader("Overlay")
        st.image(blended, use_container_width=True)
