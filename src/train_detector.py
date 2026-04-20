from pathlib import Path
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detect_dataset import LicensePlateDetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class ToTensor:
    def __call__(self, image):
        return F.to_tensor(image)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Detector] Device: {device}")

    save_dir = Path("checkpoints/detector")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = LicensePlateDetectionDataset(
        images_dir="dataset/detection/train/images",
        labels_dir="dataset/detection/train/labels",
        transforms=ToTensor(),
    )

    val_dataset = LicensePlateDetectionDataset(
        images_dir="dataset/detection/valid/images",
        labels_dir="dataset/detection/valid/labels",
        transforms=ToTensor(),
    )

    print(f"[Detector] Train samples: {len(train_dataset)}")
    print(f"[Detector] Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"[Detector] Train batches: {len(train_loader)}")
    print(f"[Detector] Val batches: {len(val_loader)}")

    model = get_model(num_classes=2)  # background + license_plate
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 1
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"[Epoch {epoch+1}] started")

        epoch_start = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if step % 10 == 0:
                elapsed = time.time() - epoch_start
                avg_step_time = elapsed / step
                remaining_steps = len(train_loader) - step
                eta_seconds = avg_step_time * remaining_steps

                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)

                print(
                    f"[Epoch {epoch+1}] Step {step}/{len(train_loader)} | "
                    f"Loss: {losses.item():.4f} | "
                    f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

            if step % 1000 == 0:
                step_path = save_dir / f"detector_epoch_{epoch+1}_step_{step}.pth"
                torch.save(model.state_dict(), step_path)
                print(f"[Step Save] {step_path}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch+1}/{num_epochs}] Average Train Loss: {avg_loss:.4f}")

        epoch_path = save_dir / f"detector_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_path)
        print(f"[Saved] {epoch_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_detector.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[Best] saved to {best_path}")

    print("[Detector] Training finished.")


if __name__ == "__main__":
    main()