import torch
import torch.nn as nn
import numpy as np
from model import UNET
from utils import load_checkpoint, get_loaders, save_predictions_as_imgs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Ustawienia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True

# Ścieżki do zbioru ewaluacyjnego
EVAL_IMG_DIR = "data/2d_images/validation"
EVAL_MASK_DIR = "data/2d_masks/validation"

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    
    dice = (2. * intersection) / (union + 1e-6)
    return dice.item()

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.item()

def evaluate_model(loader, model):
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for data, targets in loop:
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            preds = torch.sigmoid(model(data))

            dice = dice_score(preds, targets)
            iou = iou_score(preds, targets)

            dice_scores.append(dice)
            iou_scores.append(iou)

            loop.set_postfix(Dice=np.mean(dice_scores), IoU=np.mean(iou_scores))

    model.train()
    print(f"Średni Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Średni IoU Score: {np.mean(iou_scores):.4f}")

# Główna funkcja
def main():
    eval_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    eval_loader, _ = get_loaders(
        EVAL_IMG_DIR,
        EVAL_MASK_DIR,
        EVAL_IMG_DIR,
        EVAL_MASK_DIR,
        BATCH_SIZE,
        eval_transform,
        eval_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    
    # Wczytanie wytrenowanego modelu
    checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
    load_checkpoint(checkpoint, model)

    # Ewaluacja
    evaluate_model(eval_loader, model)

    save_predictions_as_imgs(eval_loader, model, folder="saved_images_validation/", device=DEVICE)

if __name__ == "__main__":
    main()