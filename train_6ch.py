"""
Entry point — trains RegNet6 (6-channel RGB pair input).

Usage:
    python train_6ch.py
"""

import torch
from src.config  import MODEL_SAVE_PATH, LR_6CH
from src.dataset import get_dataloaders
from src.model   import RegNet6
from src.train   import train_model
from src.evaluate import evaluate, plot_roc, extract_embeddings, plot_tsne, cosine_distances

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

train_loader, val_loader, test_loader, test_df = get_dataloaders(mode="rgb")

model   = RegNet6().to(DEVICE)
history = train_model(model, train_loader, val_loader,
                      save_path=MODEL_SAVE_PATH, device=DEVICE, lr=LR_6CH)

y_true, y_scores = evaluate(model, test_loader, DEVICE)
plot_roc(y_true, y_scores, title="ROC — RegNet6 (6ch RGB)")

embeddings, labels = extract_embeddings(model, test_loader, DEVICE)
plot_tsne(embeddings, labels)
cosine_distances(embeddings, labels)
