"""
Entry point — LVNN: runs the CNN on test samples and generates
LLM explanations via LLaMA 4 Maverick (OpenRouter).

Requires OPENROUTER_API_KEY in .env.

Usage:
    python explain.py
"""

import torch
import numpy as np
import pandas as pd

from src.config  import MODEL_SAVE_PATH, EMBEDDINGS_SAVE_PATH, CSV_PATH, IMAGE_FOLDER, LVNN_N_SAMPLES
from src.model   import RegNet6, load_model
from src.dataset import load_csv
from src.lvnn    import run_lvnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model      = load_model(RegNet6, MODEL_SAVE_PATH, DEVICE)
embeddings = np.load(EMBEDDINGS_SAVE_PATH)
df         = load_csv(CSV_PATH)
test_df    = df[df["conjunto"] == 2]

run_lvnn(model, test_df, embeddings, IMAGE_FOLDER, DEVICE, n_samples=LVNN_N_SAMPLES)
