# config.py
import torch
import os

# --- Basic Settings ---
RUN_NAME = "resnet_lstm_intensity_v1" # Name for this training run (affects logging/checkpoint dirs)
SEED = 42                     # Random seed for reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Settings ---
# IMPORTANT: Set this to your actual data root directory
ROOT_DIR = "/media/jacktao/Document/AU" #"D:\AU" # e.g., "/data/typhoon_dataset"
METADATA_JSON_PATH = os.path.join(ROOT_DIR, "metadata.json")
IMAGE_DIR = os.path.join(ROOT_DIR, "image")
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")

STEP_SIZE = 200 #滑动窗口的大小 
TARGET_COLUMN = "grade" # Column in CSV to predict ('grade', 'wind', 'pressure')
# TARGET_COLUMN = "wind" # Example: if predicting wind speed

# Define how to split typhoon IDs into train/validation/test sets
# Example: 80% train, 10% validation, 10% test (adjust ratios as needed)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO will be the remainder

# --- Model Settings ---
# ResNet backbone (e.g., resnet18, resnet34, resnet50)
RESNET_MODEL = "resnet18"
PRETRAINED = True          # Use ImageNet pre-trained weights for ResNet

# LSTM settings
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# --- Training Settings ---
SEQUENCE_LENGTH = 10     # Number of consecutive images in one sequence sample
BATCH_SIZE = 2          # Number of sequences per batch
NUM_EPOCHS = 3         # Maximum number of training epochs
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5      # Optimizer weight decay

# --- Dataloader Settings ---
# Set num_workers based on OS. Windows generally requires 0 for stability.
NUM_WORKERS = 0 if os.name == 'nt' else 16

# --- Checkpointing & Logging ---
CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
LOG_DIR = os.path.join("runs", RUN_NAME) # TensorBoard logs
SAVE_CHECKPOINT_EPOCH_FREQ = 1 # How often to save the 'last' checkpoint (in epochs)
BEST_CHECKPOINT_METRIC = "val_loss" # Metric to monitor for saving best model ('val_loss')
OUTPUT_DIR =os.path.join("output", RUN_NAME) # used to store result csv and txt

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 10 # Epochs to wait for improvement before stopping
EARLY_STOPPING_DELTA = 0.001 # Minimum change to qualify as improvement

# --- Data Preprocessing/Normalization ---
# Example: Normalize brightness temperature (Kelvin) to [0, 1]
# Adjust these based on your data's actual range or calculated stats
NORMALIZE = True
TEMP_MIN = 180.0 # Estimated minimum Kelvin temperature
TEMP_MAX = 310.0 # Estimated maximum Kelvin temperature

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# For classification task
NUM_CLASSES = 10  # Number of classes in your dataset
BEST_CHECKPOINT_METRIC = 'val_accuracy'  # or 'val_loss' if you prefer