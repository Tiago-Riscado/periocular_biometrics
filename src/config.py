import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
IMAGE_FOLDER         = os.getenv("IMAGE_FOLDER",         "./data/both_eyes")
CSV_PATH             = os.getenv("CSV_PATH",             "./data/output.csv")
MODEL_SAVE_PATH      = os.getenv("MODEL_SAVE_PATH",      "./models/regnet6_model.pth")
EMBEDDINGS_SAVE_PATH = os.getenv("EMBEDDINGS_SAVE_PATH", "./results/embeddings_test.npy")
LABELS_SAVE_PATH     = os.getenv("LABELS_SAVE_PATH",     "./results/labels_test.npy")

# --- Dataset generation ---
TOT_COMPARISONS   = int(os.getenv("TOT_COMPARISONS", 30000))
PROPORTIONS_LVT   = [0.7, 0.1, 0.2]   # train / val / test

# --- Training ---
BATCH_SIZE    = 16
EPOCHS        = 50
LR_6CH        = 0.001
LR_2CH        = 0.0001
PATIENCE      = 5
TARGET_SIZE   = (224, 128)

# --- ImageNet normalisation ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- LLM (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL          = "meta-llama/llama-4-maverick:free"
LLM_TEMPERATURE    = 1.2
LLM_MAX_TOKENS     = 1000
LLM_SLEEP_SECONDS  = 10
LVNN_N_SAMPLES     = 5
