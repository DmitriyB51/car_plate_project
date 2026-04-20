%%writefile src/train.py
"""
train.py - Fine-tuning TrOCR on license plate OCR dataset

Usage:
    python src/train.py
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import yaml
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import build_datasets


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model_and_processor(config: dict):
    model_name = config["model"]["name"]
    print(f"[Setup] Loading: {model_name}")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.config.max_new_tokens = config["model"]["max_target_length"]
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 1

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Setup] Model parameters: {total_params:,}")
    return processor, model


def build_compute_metrics(processor: TrOCRProcessor):
    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred

        label_ids_clean = np.where(
            label_ids != -100, label_ids, processor.tokenizer.pad_token_id
        )

        pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = processor.tokenizer.batch_decode(label_ids_clean, skip_special_tokens=True)

        pred_strs = [s.strip().upper() for s in pred_strs]
        label_strs = [s.strip().upper() for s in label_strs]

        cer = cer_metric.compute(predictions=pred_strs, references=label_strs)
        exact_acc = sum(p == g for p, g in zip(pred_strs, label_strs)) / len(pred_strs)

        return {
            "cer": round(cer, 4),
            "exact_acc": round(exact_acc, 4),
        }

    return compute_metrics


def main(config_path: str):
    config = load_config(config_path)

    processor, model = setup_model_and_processor(config)

    train_ds, val_ds, test_ds = build_datasets(
        annotations_file=str(PROJECT_ROOT / config["dataset"]["annotations_file"]),
        images_dir=str(PROJECT_ROOT / config["dataset"]["images_dir"]),
        processor=processor,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        random_seed=config["dataset"]["random_seed"],
        max_target_length=config["model"]["max_target_length"],
    )

    tc = config["training"]
    output_dir = str(PROJECT_ROOT / tc["output_dir"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        warmup_steps=tc["warmup_steps"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        save_total_limit=tc["save_total_limit"],
        save_strategy=tc["save_strategy"],
        eval_strategy=tc["eval_strategy"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=tc["greater_is_better"],
        predict_with_generate=tc["predict_with_generate"],
        generation_max_length=tc["generation_max_length"],
        logging_steps=tc["logging_steps"],
        report_to=tc["report_to"],
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )

    print("[Train] Starting fine-tuning...")
    trainer.train()

    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"[Train] Best model saved to: {best_dir}")

    print("[Train] Evaluating on test set...")
    test_results = trainer.predict(test_ds)
    test_cer = test_results.metrics.get("test_cer", "N/A")
    test_exact = test_results.metrics.get("test_exact_acc", "N/A")
    print(f"[Train] Test CER:       {test_cer}")
    print(f"[Train] Test Exact Acc: {test_exact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on license plates")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)