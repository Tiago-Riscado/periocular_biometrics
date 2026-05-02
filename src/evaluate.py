import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

from src.config import EMBEDDINGS_SAVE_PATH, LABELS_SAVE_PATH


# ------------------------------------------------------------------ #
# Test-set evaluation
# ------------------------------------------------------------------ #

def evaluate(model, test_loader, device: torch.device):
    model.eval()
    y_true, y_scores, y_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy().flatten())
            y_scores.extend(outputs.cpu().numpy().flatten())
            y_preds.extend((outputs.cpu().numpy() > 0.5).astype(int).flatten())

    acc = accuracy_score(y_true, y_preds)
    print(f"Test Accuracy: {acc:.4f}")
    return np.array(y_true), np.array(y_scores)


def plot_roc(y_true, y_scores, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Embeddings
# ------------------------------------------------------------------ #

def extract_embeddings(model, test_loader, device: torch.device,
                       save_embeddings: str = EMBEDDINGS_SAVE_PATH,
                       save_labels: str     = LABELS_SAVE_PATH):
    """
    Registers a forward hook on model.model.avgpool to capture embeddings,
    saves them as .npy files.
    """
    embeddings   = []
    labels_list  = []

    def hook(module, input, output):
        pooled = output.detach().cpu()
        if pooled.ndim > 2:
            pooled = nn.functional.adaptive_avg_pool2d(pooled, (1, 1)).squeeze()
        embeddings.append(pooled)

    handle = model.model.avgpool.register_forward_hook(hook)
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            _ = model(inputs.to(device))
            labels_list.extend(labels.numpy())

    handle.remove()

    # Align lengths (safety)
    n = min(len(embeddings), len(labels_list))
    embeddings  = embeddings[:n]
    labels_list = labels_list[:n]

    matrix = torch.stack(embeddings).view(n, -1).numpy()
    np.save(save_embeddings, matrix)
    np.save(save_labels, np.array(labels_list))
    print(f"✓ Embeddings saved to {save_embeddings}")
    print(f"✓ Labels saved to {save_labels}")
    return matrix, labels_list


def plot_tsne(embeddings_matrix, labels_list):
    if embeddings_matrix.shape[0] <= 1:
        print("Not enough embeddings for t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    emb2d = tsne.fit_transform(embeddings_matrix)

    df = pd.DataFrame(emb2d, columns=["Dim1", "Dim2"])
    df["Class"] = labels_list

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Class",
                    palette="Set1", alpha=0.7)
    plt.title("t-SNE of Embeddings by Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def cosine_distances(embeddings_matrix, labels_list):
    positives, negatives = [], []

    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            d = cosine(embeddings_matrix[i], embeddings_matrix[j])
            (positives if labels_list[i] == labels_list[j] else negatives).append(d)

    print(f"Avg cosine distance — Genuine:  {np.mean(positives):.4f}")
    print(f"Avg cosine distance — Impostor: {np.mean(negatives):.4f}")
