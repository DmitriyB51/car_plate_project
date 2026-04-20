import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class LicensePlateDataset(Dataset):
    """
    PyTorch Dataset for license plate OCR.

    Adds light augmentation only for training data.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        images_dir: str,
        processor: TrOCRProcessor,
        max_target_length: int = 20,
        is_train: bool = False,
    ):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.30:
            angle = random.uniform(-3, 3)
            image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

        if random.random() < 0.30:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < 0.30:
            factor = random.uniform(0.85, 1.20)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() < 0.20:
            factor = random.uniform(0.8, 1.4)
            image = ImageEnhance.Sharpness(image).enhance(factor)

        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename, plate_text = self.samples[idx]
        image_path = self.images_dir / filename

        image = Image.open(image_path).convert("RGB")

        if self.is_train:
            image = self._augment_image(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            plate_text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def load_annotations(annotations_file: str) -> List[Tuple[str, str]]:
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations: Dict[str, str] = json.load(f)

    samples = []
    for filename, plate_text in annotations.items():
        cleaned = plate_text.strip().upper()
        if cleaned:
            samples.append((filename, cleaned))

    return samples


def create_splits(
    samples: List[Tuple[str, str]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    random.seed(random_seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_s = shuffled[:train_end]
    val_s = shuffled[train_end:val_end]
    test_s = shuffled[val_end:]

    print(f"[Dataset] Total: {n} | Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")
    return train_s, val_s, test_s


def build_datasets(
    annotations_file: str,
    images_dir: str,
    processor: TrOCRProcessor,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    random_seed: int = 42,
    max_target_length: int = 20,
) -> Tuple["LicensePlateDataset", "LicensePlateDataset", "LicensePlateDataset"]:
    all_samples = load_annotations(annotations_file)
    train_s, val_s, test_s = create_splits(all_samples, train_ratio, val_ratio, random_seed)

    train_ds = LicensePlateDataset(
        train_s,
        images_dir,
        processor,
        max_target_length,
        is_train=True,
    )
    val_ds = LicensePlateDataset(
        val_s,
        images_dir,
        processor,
        max_target_length,
        is_train=False,
    )
    test_ds = LicensePlateDataset(
        test_s,
        images_dir,
        processor,
        max_target_length,
        is_train=False,
    )

    return train_ds, val_ds, test_ds