import sys
from pathlib import Path

import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import LicensePlatePredictor, load_config  # noqa: E402

st.set_page_config(
    page_title="Kazakh License Plate OCR",
    page_icon="🚗",
    layout="centered"
)

st.title("Kazakh License Plate OCR")
st.write("Upload a car photo — the app will detect the license plate and read it automatically.")

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DETECTOR_CKPT = PROJECT_ROOT / "checkpoints" / "detector" / "best_detector.pth"


def _build_detector(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@st.cache_resource
def load_models():
    config = load_config(str(CONFIG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = _build_detector()
    detector.load_state_dict(torch.load(str(DETECTOR_CKPT), map_location=device))
    detector.to(device)
    detector.eval()

    checkpoint_path = PROJECT_ROOT / config["inference"]["checkpoint_path"]
    if not checkpoint_path.exists():
        checkpoint_folders = sorted(
            [p for p in (PROJECT_ROOT / "checkpoints").glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoint_folders:
            checkpoint_path = checkpoint_folders[-1]
        else:
            raise FileNotFoundError("No OCR checkpoint found in checkpoints/")

    ocr = LicensePlatePredictor(
        checkpoint_path=str(checkpoint_path),
        num_beams=config["inference"]["num_beams"],
        max_new_tokens=config["model"]["max_target_length"],
    )
    return detector, ocr, device


uploaded_file = st.file_uploader(
    "Upload car image",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    score_threshold = st.slider("Detection confidence threshold", 0.1, 1.0, 0.7, 0.05)

    if st.button("Detect & Recognize"):
        with st.spinner("Loading models..."):
            try:
                detector, ocr, device = load_models()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.stop()

        with st.spinner("Detecting license plate..."):
            tensor = F.to_tensor(image).to(device)
            with torch.no_grad():
                prediction = detector([tensor])[0]

            boxes = prediction["boxes"]
            scores = prediction["scores"]

            best_box, best_score = None, -1.0
            for box, score in zip(boxes, scores):
                s = float(score.item())
                if s >= score_threshold and s > best_score:
                    best_score = s
                    best_box = box

        if best_box is None:
            st.warning(f"No plate detected with confidence >= {score_threshold}. Try lowering the threshold.")
        else:
            x1, y1, x2, y2 = best_box.int().cpu().numpy()
            plate_crop = image.crop((x1, y1, x2, y2))

            with st.spinner("Running OCR..."):
                plate_text = ocr.predict(plate_crop)

            st.success("Done!")
            col1, col2 = st.columns(2)
            with col1:
                st.image(plate_crop, caption="Detected plate crop")
            with col2:
                st.metric("Recognized plate", plate_text)
                st.caption(f"Detection confidence: {best_score:.2%}")
                st.caption(f"Plate coordinates: ({x1}, {y1}) — ({x2}, {y2})")
