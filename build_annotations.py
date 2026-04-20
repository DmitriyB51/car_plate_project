%%writefile build_annotations.py
import os
import json

DATASET_ROOT = r"C:\Users\Mukhamedali\Downloads\autoriaNumberplateOcrKz-2019-04-26\autoriaNumberplateOcrKz-2019-04-26"
PROJECT_DATASET = r"C:\Users\Mukhamedali\Desktop\car_plate_project\dataset"
IMAGES_DIR = os.path.join(PROJECT_DATASET, "images")
OUTPUT_JSON = os.path.join(PROJECT_DATASET, "annotations.json")

annotations = {}

image_files = set(os.listdir(IMAGES_DIR))

for split in ["train", "val", "test"]:
    ann_dir = os.path.join(DATASET_ROOT, split, "ann")

    for fname in os.listdir(ann_dir):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(ann_dir, fname)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data["name"]
        text = data["description"].strip().upper()

        possible_names = [
            f"{name}.png",
            f"{name}.jpg",
            f"{name}.jpeg",
            f"{name}.bmp",
            f"{name}.webp"
        ]

        real_image_name = None
        for candidate in possible_names:
            if candidate in image_files:
                real_image_name = candidate
                break

        if real_image_name is None:
            print(f"[WARNING] Image not found for: {name}")
            continue

        annotations[real_image_name] = text

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"Done. Saved {len(annotations)} annotations to {OUTPUT_JSON}")