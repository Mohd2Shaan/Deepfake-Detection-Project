"""Streamlit demo: upload a face image -> Real/Fake prediction + confidence + Grad-CAM.

    streamlit run app.py
"""

import streamlit as st
from PIL import Image

from src import config
from src.gradcam import GradCAMExplainer, to_display_rgb
from src.model import load_checkpoint

st.set_page_config(page_title="Deepfake Detection")


@st.cache_resource
def load_explainer(checkpoint_path):
    device = config.get_device()
    model, ckpt = load_checkpoint(checkpoint_path, device)
    return GradCAMExplainer(model, ckpt["model_name"], device), ckpt


st.title("Deepfake Detection")
st.write("Upload a face image to check whether it is a real photo or a StyleGAN-generated fake.")

checkpoints = sorted(config.CHECKPOINT_DIR.glob("*_best.pth"))
if not checkpoints:
    st.error("No trained model found in checkpoints/. Train one first: python -m src.train")
    st.stop()

checkpoint = st.selectbox("Model", checkpoints, format_func=lambda p: p.name)
uploaded = st.file_uploader("Face image", type=["jpg", "jpeg", "png", "webp"])
show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    explainer, ckpt = load_explainer(str(checkpoint))

    with st.spinner("Running model..."):
        result = explainer.explain(image)

    label = config.CLASS_NAMES[result["pred"]].upper()
    confidence = result["prob_fake"] if result["pred"] == 1 else 1 - result["prob_fake"]

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
    st.progress(float(result["prob_fake"]), text=f"P(fake) = {result['prob_fake']:.4f}")

    col1, col2 = st.columns(2)
    col1.image(to_display_rgb(image), caption="Input (resized to 224x224)", width="stretch")
    if show_heatmap:
        col2.image(result["overlay"], caption="Grad-CAM: regions that influenced the prediction",
                   width="stretch")

    st.caption(f"Model: {ckpt['model_name']} | Note: trained only on StyleGAN faces - "
               "results on other kinds of fakes (diffusion, face swaps) may be unreliable.")
