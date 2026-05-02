import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from src.config import (
    IMAGE_FOLDER, CSV_PATH, TARGET_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE,
    TOT_COMPARISONS, PROPORTIONS_LVT,
)


# ------------------------------------------------------------------ #
# CSV generation
# ------------------------------------------------------------------ #

def _generate_genuine(L):
    idx1 = np.random.randint(0, len(L))
    while True:
        idx2 = np.random.randint(0, len(L))
        if L[idx1][1] == L[idx2][1] and idx1 != idx2:
            return [L[idx1][0], L[idx2][0]]


def _generate_impostor(L):
    idx1 = np.random.randint(0, len(L))
    while True:
        idx2 = np.random.randint(0, len(L))
        if L[idx1][1] != L[idx2][1]:
            return [L[idx1][0], L[idx2][0]]


def generate_csv(image_folder: str = IMAGE_FOLDER,
                 output_path: str = CSV_PATH,
                 tot: int = TOT_COMPARISONS,
                 proportions: list = PROPORTIONS_LVT):
    """
    Scans image_folder for periocular images (pattern C*_S*_I*.jpg),
    splits S1 → learn/val and S2 → test, generates genuine/impostor
    pairs and writes them to output_path.
    """
    pattern = r"C(\d+)_S(\d+)_I(\d+)\.jpg"
    LV, T = [], []

    for f in os.listdir(image_folder):
        if not f.endswith(".jpg"):
            continue
        m = re.match(pattern, f)
        if not m:
            continue
        entry = [f, int(m.group(1)), int(m.group(2)), int(m.group(3))]
        (LV if "S1" in f else T).append(entry)

    comparisons = []
    for split_list, prop, split_id in [(LV, proportions[0], 0),
                                        (LV, proportions[1], 1),
                                        (T,  proportions[2], 2)]:
        n = int(tot * prop / 2)
        for _ in range(n):
            comparisons.append([_generate_genuine(split_list),  1, split_id])
            comparisons.append([_generate_impostor(split_list), 0, split_id])

    with open(output_path, "w") as f:
        for item in comparisons:
            f.write("%s, %s, %d, %d \n" % (item[0][0], item[0][1], item[1], item[2]))

    print(f"✓ {len(comparisons)} pairs written to {output_path}")


# ------------------------------------------------------------------ #
# Transforms
# ------------------------------------------------------------------ #

def get_transform_rgb():
    return transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transform_gray():
    return transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[IMAGENET_MEAN[0]], std=[IMAGENET_STD[0]]),
    ])


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class FaceDataset(Dataset):
    """
    Loads pairs of periocular images and concatenates them channel-wise.
    mode='rgb'  → 6-channel tensor  (RGB + RGB)
    mode='gray' → 2-channel tensor  (L + L)
    """

    def __init__(self, dataframe: pd.DataFrame, image_folder: str,
                 mode: str = "rgb"):
        self.df           = dataframe
        self.image_folder = image_folder
        self.mode         = mode
        self.transform    = get_transform_rgb() if mode == "rgb" else get_transform_gray()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p1  = os.path.join(self.image_folder, row["img1"].strip())
        p2  = os.path.join(self.image_folder, row["img2"].strip())

        convert = "RGB" if self.mode == "rgb" else "L"
        img1 = Image.open(p1).convert(convert)
        img2 = Image.open(p2).convert(convert)

        return torch.cat([self.transform(img1),
                          self.transform(img2)], dim=0), int(row["classe"])


# ------------------------------------------------------------------ #
# DataLoaders
# ------------------------------------------------------------------ #

def load_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.columns = ["img1", "img2", "classe", "conjunto"]
    return df[df["img1"].str.endswith(".jpg") & df["img2"].str.endswith(".jpg")]


def get_dataloaders(mode: str = "rgb", batch_size: int = BATCH_SIZE):
    df         = load_csv()
    train_df   = df[df["conjunto"] == 0]
    val_df     = df[df["conjunto"] == 1]
    test_df    = df[df["conjunto"] == 2]

    def _loader(split_df, shuffle):
        ds = FaceDataset(split_df, IMAGE_FOLDER, mode=mode)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _loader(train_df, shuffle=True),
        _loader(val_df,   shuffle=False),
        _loader(test_df,  shuffle=False),
        test_df,
    )
