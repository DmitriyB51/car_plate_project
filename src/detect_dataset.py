from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


class LicensePlateDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, bw, bh = map(float, parts)

                    x_center *= w
                    y_center *= h
                    bw *= w
                    bh *= h

                    x_min = x_center - bw / 2
                    y_min = y_center - bh / 2
                    x_max = x_center + bw / 2
                    y_max = y_center + bh / 2

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target