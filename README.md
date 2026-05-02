# Periocular Biometric Verification

Deep learning system for periocular identity verification using pairwise image comparison, with LLM-generated explanations (LVNN).

## Structure

```
periocular-biometrics/
│
├── generate_csv.py     # Step 1 — generate genuine/impostor pairs CSV
├── train_6ch.py        # Step 2a — train RegNet6 (RGB, 6-channel)
├── train_2ch.py        # Step 2b — train RegNet2 (grayscale, 2-channel)
├── explain.py          # Step 3 — LVNN: CNN + LLaMA explanations
│
├── src/
│   ├── config.py       # All paths and hyperparameters (reads .env)
│   ├── dataset.py      # CSV generation, FaceDataset, DataLoaders
│   ├── model.py        # RegNet6 and RegNet2 architectures
│   ├── train.py        # Shared training loop with early stopping
│   ├── evaluate.py     # ROC curve, embeddings, t-SNE, cosine distance
│   └── lvnn.py         # LLM prompt builder + OpenRouter API call
│
├── data/
│   ├── both_eyes/      # Periocular image dataset (not versioned)
│   └── output.csv      # Generated pair comparisons
│
├── models/             # Saved .pth weights (not versioned)
├── results/            # Saved embeddings .npy (not versioned)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

## Pipeline

```
images/ → generate_csv.py → output.csv
                                ↓
              train_6ch.py / train_2ch.py
                                ↓
              regnet6_model.pth + embeddings_test.npy
                                ↓
                          explain.py (LVNN)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in paths and API key
```

## Usage

```bash
# 1. Generate pair CSV from image folder
python generate_csv.py

# 2a. Train 6-channel RGB model (+ extract embeddings + t-SNE)
python train_6ch.py

# 2b. Train 2-channel grayscale model
python train_2ch.py

# 3. Run LVNN explanations (requires OPENROUTER_API_KEY)
python explain.py
```

## Models

| Model | Input | Channels | Architecture |
|-------|-------|----------|-------------|
| RegNet6 | RGB pair | 6 | RegNet-Y-400MF |
| RegNet2 | Grayscale pair | 2 | RegNet-Y-400MF |

## Dataset format

Images follow the naming pattern `C{id}_S{session}_I{index}.jpg`:
- `S1` → learn / validation split
- `S2` → test split

The CSV has columns: `img1, img2, classe (0/1), conjunto (0/1/2)`.

## Tech stack

`PyTorch` `torchvision` `scikit-learn` `scipy` `OpenRouter (LLaMA 4)`
