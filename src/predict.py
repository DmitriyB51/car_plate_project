"""
predict.py - Single-image inference with trained TrOCR model

Usage:
    python src/predict.py --image path/to/plate.jpg
    python src/predict.py --image path/to/plate.jpg --checkpoint checkpoints/best_model
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LicensePlatePredictor:
    """
    Reusable predictor. Load once, call predict() many times.
    Suitable for batch processing or serving in a loop.
    """

    def __init__(self, checkpoint_path: str, num_beams: int = 4, max_new_tokens: int = 20):
        print(f"[Predictor] Loading model from: {checkpoint_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Processor берем из базовой модели TrOCR
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

        # Веса модели берем из обученного checkpoint
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()

        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        print("[Predictor] Ready.")

    def predict(self, image_input) -> str:
        """
        Predict plate text from an image.

        Args:
            image_input: str path, Path, or PIL.Image.Image

        Returns:
            Predicted plate text (uppercased)
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                early_stopping=True,
            )

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip().upper()

    def predict_batch(self, image_paths: list) -> list:
        """Predict plate text for a list of image paths."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                early_stopping=True,
            )

        texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip().upper() for t in texts]


def main(image_path: str, config_path: str, checkpoint_path: str = None):
    config = load_config(config_path)

    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / config["inference"]["checkpoint_path"])

    predictor = LicensePlatePredictor(
        checkpoint_path = checkpoint_path,
        num_beams       = config["inference"]["num_beams"],
        max_new_tokens  = config["model"]["max_target_length"],
    )

    result = predictor.predict(image_path)
    print(f"\n[Predict] Image:      {image_path}")
    print(f"[Predict] Plate text: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict license plate text from image")
    parser.add_argument("--image",      type=str, required=True,  help="Path to plate image")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config")
    args = parser.parse_args()
    main(args.image, args.config, args.checkpoint)
