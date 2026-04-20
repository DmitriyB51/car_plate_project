from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path("checkpoints/detector/best_detector.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    test_dir = Path("dataset/detection/test/images")
    image_files = sorted([
        p for p in test_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    if not image_files:
        raise FileNotFoundError("No test images found in dataset/detection/test/images")

    image_path = image_files[0]
    print(f"[Eval] Using image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]

    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    shown = 0
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score < 0.8:
            continue

        x1, y1, x2, y2 = box.int().cpu().numpy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        shown += 1

    print(f"[Eval] Boxes shown with score >= 0.5: {shown}")

    output_path = Path("detector_eval_result.jpg")
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[Eval] Saved result to: {output_path}")

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()