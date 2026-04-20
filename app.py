import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import LicensePlatePredictor, load_config  # noqa: E402


st.set_page_config(
    page_title="Kazakh License Plate OCR",
    page_icon="🚗",
    layout="centered"
)

st.title("Kazakh License Plate OCR")
st.write("Upload a cropped license plate image and get the recognized text.")

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@st.cache_resource
def load_predictor():
    config = load_config(str(CONFIG_PATH))

    checkpoint_path = PROJECT_ROOT / config["inference"]["checkpoint_path"]

    # если checkpoints/best_model не существует, пробуем взять последний checkpoint-*
    if not checkpoint_path.exists():
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_folders = sorted(
            [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoint_folders:
            checkpoint_path = checkpoint_folders[-1]
        else:
            raise FileNotFoundError(
                "No valid checkpoint found. Put a model into checkpoints/best_model "
                "or keep at least one checkpoints/checkpoint-* folder."
            )

    predictor = LicensePlatePredictor(
        checkpoint_path=str(checkpoint_path),
        num_beams=config["inference"]["num_beams"],
        max_new_tokens=config["model"]["max_target_length"],
    )
    return predictor, checkpoint_path


uploaded_file = st.file_uploader(
    "Upload plate image",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Recognize"):
        with st.spinner("Running OCR..."):
            try:
                predictor, checkpoint_path = load_predictor()
                prediction = predictor.predict(image)

                st.success("Recognition complete")
                st.write(f"**Predicted plate text:** `{prediction}`")
                st.caption(f"Model checkpoint: {checkpoint_path}")

            except Exception as e:
                st.error(f"Error: {e}")