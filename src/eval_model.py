"""
eval_model.py - Detailed evaluation of a trained TrOCR checkpoint
"""

import argparse
import sys
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import LicensePlateDataset, build_datasets


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluation(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    dataset: LicensePlateDataset,
    config: dict,
    num_beams: int = 4,
    num_samples_to_print: int = 10,
) -> dict:
    model.eval()
    model.to(torch.device("cpu"))

    cer_metric = hf_evaluate.load("cer")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_preds = []
    all_labels = []

    print(f"[Evaluate] {len(dataset)} samples, beam_size={num_beams}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"]
            label_ids = batch["labels"]

            generated_ids = model.generate(
                pixel_values,
                num_beams=num_beams,
                max_new_tokens=config["model"]["max_target_length"],
                early_stopping=True,
            )

            pred_strs = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            label_ids_clean = np.where(
                label_ids.numpy() != -100,
                label_ids.numpy(),
                processor.tokenizer.pad_token_id,
            )
            gt_strs = processor.batch_decode(
                label_ids_clean, skip_special_tokens=True
            )

            all_preds.extend(s.strip().upper() for s in pred_strs)
            all_labels.extend(s.strip().upper() for s in gt_strs)

    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    exact_matches = sum(p == g for p, g in zip(all_preds, all_labels))
    exact_acc = exact_matches / len(all_preds)

    print(f"\n{'Ground Truth':<20} {'Prediction':<20} {'Match'}")
    print("-" * 50)
    for gt, pred in zip(all_labels[:num_samples_to_print], all_preds[:num_samples_to_print]):
        status = "OK" if gt == pred else "MISMATCH"
        print(f"{gt:<20} {pred:<20} {status}")

    print(f"\n[Results]")
    print(f"  CER (Character Error Rate): {cer:.4f}  (0.0 = perfect)")
    print(f"  Exact Match Accuracy:       {exact_acc:.4f}  ({exact_matches}/{len(all_preds)} plates correct)")
    print(f"  Total evaluated:            {len(all_preds)}")

    return {
        "cer": cer,
        "exact_acc": exact_acc,
        "predictions": all_preds,
        "ground_truths": all_labels,
    }


def main(config_path: str, checkpoint_path: str = None, num_samples: int = 10):
    config = load_config(config_path)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / config["inference"]["checkpoint_path"]
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        checkpoint_folders = sorted(
            [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoint_folders:
            checkpoint_path = checkpoint_folders[-1]
            print(f"[Evaluate] Falling back to latest checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No valid checkpoint found: {checkpoint_path}")

    print(f"[Evaluate] Loading checkpoint: {checkpoint_path}")

    # Processor always from base model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)

    _, _, test_ds = build_datasets(
        annotations_file=str(PROJECT_ROOT / config["dataset"]["annotations_file"]),
        images_dir=str(PROJECT_ROOT / config["dataset"]["images_dir"]),
        processor=processor,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        random_seed=config["dataset"]["random_seed"],
        max_target_length=config["model"]["max_target_length"],
    )

    run_evaluation(
        model,
        processor,
        test_ds,
        config,
        num_beams=config["inference"]["num_beams"],
        num_samples_to_print=num_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TrOCR checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sample predictions to print")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.num_samples)