"""Train the card-segmentation model.

Run:  python train.py --config config.yaml
Saves the best model (by validation IoU) to checkpoints/best.pt."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from dataset import CardDataset, eval_tf, make_splits, train_tf
from metrics import binarize, dice_score, iou_score
from model import build_model


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_items, val_items, test_items, info = make_splits(cfg)
    print("Split by dive:", info)
    print(f"Images -> train {len(train_items)}, val {len(val_items)}, test {len(test_items)}")
    if len(train_items) == 0 or len(val_items) == 0:
        raise RuntimeError("Need at least one dive in train and one in val. Add more data "
                           "or set val_dives/test_dives in config.yaml.")

    h, w = cfg["train"]["img_height"], cfg["train"]["img_width"]
    nw = cfg["train"]["num_workers"]
    train_loader = DataLoader(CardDataset(train_items, train_tf(h, w)),
                              batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=nw, drop_last=False)
    val_loader = DataLoader(CardDataset(val_items, eval_tf(h, w)),
                            batch_size=1, shuffle=False, num_workers=nw)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    # Loss = pixel-wise BCE + Dice. BCE gets every pixel roughly right; Dice cares
    # about overlap of the card shape, which is what IoU rewards.
    bce = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")

    os.makedirs(cfg["output"]["ckpt_dir"], exist_ok=True)
    best_iou, bad_epochs = -1.0, 0
    patience = cfg["train"]["early_stop_patience"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(train_loader.dataset))

        # Validation
        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = binarize(model(x))
                ious.append(iou_score(pred, y))
                dices.append(dice_score(pred, y))
        val_iou = sum(ious) / len(ious)
        val_dice = sum(dices) / len(dices)
        print(f"epoch {epoch + 1:3d} | train loss {train_loss:.3f} | "
              f"val IoU {val_iou:.3f} | val Dice {val_dice:.3f}")

        if val_iou > best_iou:
            best_iou, bad_epochs = val_iou, 0
            path = os.path.join(cfg["output"]["ckpt_dir"], "best.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "val_iou": best_iou}, path)
            print(f"          saved new best -> {path}  (val IoU {best_iou:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no val improvement for {patience} epochs.")
                break

    print(f"Done. Best validation IoU: {best_iou:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        run(yaml.safe_load(f))
