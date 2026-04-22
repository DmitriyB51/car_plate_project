from pathlib import Path
import sys
import csv

import cv2
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


def find_latest_ocr_checkpoint(project_root: Path, config: dict) -> Path:
    ocr_ckpt = project_root / config["inference"]["checkpoint_path"]
    if ocr_ckpt.exists():
        return ocr_ckpt

    checkpoints_dir = project_root / "checkpoints"
    checkpoint_folders = sorted(
        [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1])
    )
    if checkpoint_folders:
        return checkpoint_folders[-1]

    raise FileNotFoundError("No OCR checkpoint found.")


def main():
    config = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))

    input_dir = input("Enter folder path with car images: ").strip()
    score_threshold = 0.7

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    detector_ckpt = PROJECT_ROOT / "checkpoints" / "detector" / "best_detector.pth"
    if not detector_ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_ckpt}")

    ocr_ckpt = find_latest_ocr_checkpoint(PROJECT_ROOT, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = load_detector(str(detector_ckpt), device)
    ocr = LicensePlatePredictor(
        checkpoint_path=str(ocr_ckpt),
        num_beams=config["inference"]["num_beams"],
        max_new_tokens=config["model"]["max_target_length"],
    )

    output_dir = PROJECT_ROOT / "batch_results"
    annotated_dir = output_dir / "annotated"
    crops_dir = output_dir / "crops"
    output_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    image_files = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    csv_path = output_dir / "results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "detected", "score", "plate_text", "bbox"])

        for idx, image_path in enumerate(image_files, start=1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

            image_pil = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(image_pil).to(device)

            with torch.no_grad():
                prediction = detector([tensor])[0]

            best_box, best_score = get_best_plate_box(prediction, score_threshold=score_threshold)

            img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

            if best_box is None:
                writer.writerow([image_path.name, 0, "", "", ""])
                out_path = annotated_dir / image_path.name
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print("  -> no plate detected")
                continue

            plate_crop, (x1, y1, x2, y2) = crop_plate_from_box(image_pil, best_box)
            plate_text = ocr.predict(plate_crop)

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

            annotated_path = annotated_dir / image_path.name
            crop_path = crops_dir / image_path.name

            cv2.imwrite(str(annotated_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            plate_crop.save(crop_path)

            writer.writerow([
                image_path.name,
                1,
                f"{best_score:.4f}",
                plate_text,
                f"{x1},{y1},{x2},{y2}"
            ])

            print(f"  -> {plate_text} | score={best_score:.4f}")

    print(f"\nDone.")
    print(f"CSV saved to: {csv_path}")
    print(f"Annotated images: {annotated_dir}")
    print(f"Crops: {crops_dir}")


if __name__ == "__main__":
    main()