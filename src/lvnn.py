import time
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import (
    OPENROUTER_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    LLM_MAX_TOKENS, LLM_SLEEP_SECONDS, TARGET_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ------------------------------------------------------------------ #
# Image loading
# ------------------------------------------------------------------ #

_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
])


def load_image_pair(img1_path: str, img2_path: str, device: torch.device):
    """Loads an RGB pair, normalises, and returns a 6-channel tensor."""
    img1 = _transform(Image.open(img1_path).convert("RGB"))
    img2 = _transform(Image.open(img2_path).convert("RGB"))
    combined = torch.cat([img1, img2], dim=0)

    mean = torch.tensor(IMAGENET_MEAN * 2).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD  * 2).view(-1, 1, 1)
    combined = (combined - mean) / std

    return combined.unsqueeze(0).to(device)


def show_pair(img1_path: str, img2_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(Image.open(img1_path)); ax1.axis("off"); ax1.set_title("Image 1")
    ax2.imshow(Image.open(img2_path)); ax2.axis("off"); ax2.set_title("Image 2")
    plt.show()


# ------------------------------------------------------------------ #
# LLM prompt & call
# ------------------------------------------------------------------ #

def build_prompt(prob: float, embedding: np.ndarray, label: int) -> str:
    context = "Positive match" if prob > 0.5 else "No match"
    sample  = np.round(embedding[:5], 4).tolist()

    return f"""You are an expert in ocular biometrics. Explain the result of a biometric verification performed on periocular image pairs.

Consider the following data:
- Network confidence: {prob:.4f}
- Embedding vector (summary): {sample}
- Ground truth: {context} ({'label=1' if label else 'label=0'})

Base your explanation on the anatomical visual characteristics of the periocular region and the meaning of the embedding vector and confidence provided. Be clear and direct.

State whether the images belong to the same person and explain why."""


def get_llm_explanation(prompt: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       LLM_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens":  LLM_MAX_TOKENS,
        },
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def run_lvnn(model, test_df, embeddings_matrix: np.ndarray,
             image_folder: str, device: torch.device,
             n_samples: int = 5):
    """
    Samples n_samples pairs from test_df, runs the CNN,
    and prints LLM explanations.
    """
    import os
    sampled = test_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    for i, row in sampled.iterrows():
        img1 = os.path.join(image_folder, row["img1"].strip())
        img2 = os.path.join(image_folder, row["img2"].strip())
        label = int(row["classe"])

        inputs = load_image_pair(img1, img2, device)
        with torch.no_grad():
            prob = model(inputs).item()

        show_pair(img1, img2)
        prompt = build_prompt(prob, embeddings_matrix[i], label)

        print("\n===== LVNN EXPLANATION (LLaMA 4 Maverick) =====")
        try:
            print(get_llm_explanation(prompt))
        except RuntimeError as e:
            print(f"LLM call failed: {e}")

        time.sleep(LLM_SLEEP_SECONDS)
