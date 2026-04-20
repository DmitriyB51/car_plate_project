from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import LicensePlatePredictor, load_config


def get_detector(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(checkpoint_path: str, device: torch.device):
    model = get_detector(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_best_plate_box(prediction, score_threshold=0.7):
    boxes = prediction["boxes"]
    scores = prediction["scores"]

    best_box = None
    best_score = -1.0

    for box, score in zip(boxes, scores):
        s = float(score.item())
        if s < score_threshold:
            continue
        if s > best_score:
            best_score = s
            best_box = box

    return best_box, best_score


def crop_plate_from_box(image_pil: Image.Image, box):
    x1, y1, x2, y2 = box.int().cpu().numpy()
    return image_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def main():
    config = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))

    image_path = input("Enter path to car image: ").strip()
    detector_ckpt = PROJECT_ROOT / "checkpoints" / "detector" / "best_detector.pth"

    if not detector_ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_ckpt}")

    ocr_ckpt = PROJECT_ROOT / config["inference"]["checkpoint_path"]
    if not ocr_ckpt.exists():
        checkpoint_folders = sorted(
            [p for p in (PROJECT_ROOT / "checkpoints").glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoint_folders:
            ocr_ckpt = checkpoint_folders[-1]
        else:
            raise FileNotFoundError("No OCR checkpoint found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = load_detector(str(detector_ckpt), device)
    ocr = LicensePlatePredictor(
        checkpoint_path=str(ocr_ckpt),
        num_beams=config["inference"]["num_beams"],
        max_new_tokens=config["model"]["max_target_length"],
    )

    image_pil = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image_pil).to(device)

    with torch.no_grad():
        prediction = detector([tensor])[0]

    best_box, best_score = get_best_plate_box(prediction, score_threshold=0.7)

    if best_box is None:
        print("No plate detected with score >= 0.7")
        return

    plate_crop, (x1, y1, x2, y2) = crop_plate_from_box(image_pil, best_box)
    plate_text = ocr.predict(plate_crop)

    print(f"[Detection score] {best_score:.4f}")
    print(f"[Recognized plate] {plate_text}")

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{plate_text} ({best_score:.2f})",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    output_path = PROJECT_ROOT / "full_pipeline_result.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plate_crop.save(PROJECT_ROOT / "detected_plate_crop.jpg")

    print(f"Saved result image to: {output_path}")
    print(f"Saved plate crop to: {PROJECT_ROOT / 'detected_plate_crop.jpg'}")

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.imshow(plate_crop)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()