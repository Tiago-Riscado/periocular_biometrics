"""
Entry point — trains RegNet2 (2-channel grayscale pair input).

Usage:
    python train_2ch.py
"""

import torch
from src.config  import LR_2CH
from src.dataset import get_dataloaders
from src.model   import RegNet2
from src.train   import train_model
from src.evaluate import evaluate, plot_roc

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_2 = "./models/regnet2_model.pth"
print("Device:", DEVICE)

train_loader, val_loader, test_loader, _ = get_dataloaders(mode="gray")

model   = RegNet2().to(DEVICE)
history = train_model(model, train_loader, val_loader,
                      save_path=MODEL_PATH_2, device=DEVICE, lr=LR_2CH)

y_true, y_scores = evaluate(model, test_loader, DEVICE)
plot_roc(y_true, y_scores, title="ROC — RegNet2 (2ch Grayscale)")
