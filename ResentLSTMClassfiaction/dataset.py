# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import logging
from typing import List, Tuple, Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_image(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalizes image data to the [0, 1] range."""
    # Clip values to prevent outliers from dominating the range
    image = np.clip(image, min_val, max_val)
    # Perform min-max scaling
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image.astype(np.float32) # Ensure float32 for PyTorch

class TyphoonSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading sequences of typhoon satellite images and metadata.
    """
    def __init__(self,
                 root_dir: str,
                 typhoon_ids: List[str],
                 metadata_dir: str,
                 image_dir: str,
                 sequence_length: int,
                 target_column: str,
                 normalize: bool = False,
                 step_size: int = 1,
                 temp_min: float = 180.0,
                 temp_max: float = 310.0):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            typhoon_ids (List[str]): List of typhoon IDs to include in this dataset split.
            metadata_dir (str): Path to the directory containing CSV metadata files.
            image_dir (str): Path to the directory containing H5 image files.
            sequence_length (int): The length of image sequences to generate.
            target_column (str): The name of the column in the CSV to use as the target label.
            normalize (bool): Whether to normalize the image data.
            temp_min (float): Minimum temperature for normalization.
            temp_max (float): Maximum temperature for normalization.
        """
        self.root_dir = root_dir
        self.typhoon_ids = typhoon_ids
        self.metadata_dir = metadata_dir
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.normalize = normalize
        self.temp_min = temp_min
        self.temp_max = temp_max

        self.sequences = self._create_sequences(step_size)
        logging.info(f"Created {len(self.sequences)} sequences for {len(typhoon_ids)} typhoons.")

    def _create_sequences(self,step_size=5) -> List[Tuple[List[str], List[float]]]:
        """
        Reads metadata for each typhoon, validates image existence, and creates sequences.
        """
        all_sequences = []
        skipped_missing_images = 0
        skipped_short_typhoons = 0

        for typhoon_id in self.typhoon_ids:
            csv_path = os.path.join(self.metadata_dir, f"{typhoon_id}.csv")
            typhoon_image_dir = os.path.join(self.image_dir, typhoon_id)

            if not os.path.exists(csv_path):
                logging.warning(f"Metadata CSV not found for typhoon {typhoon_id}. Skipping.")
                continue
            if not os.path.isdir(typhoon_image_dir):
                 logging.warning(f"Image directory not found for typhoon {typhoon_id}. Skipping.")
                 continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logging.error(f"Error reading CSV {csv_path}: {e}. Skipping typhoon {typhoon_id}.")
                continue

            if self.target_column not in df.columns:
                logging.error(f"Target column '{self.target_column}' not found in {csv_path}. Skipping typhoon {typhoon_id}.")
                continue

            # Keep track of valid image paths and corresponding targets for this typhoon
            valid_image_paths = []
            valid_targets = []

            for _, row in df.iterrows():
                image_filename = row.get('file_1') # Use .get for safety
                if pd.isna(image_filename):
                    skipped_missing_images +=1
                    continue # Skip row if image filename is missing in CSV

                image_path = os.path.join(typhoon_image_dir, image_filename)

                # --- Critical Check: Ensure image file exists ---
                if not os.path.exists(image_path):
                    # logging.debug(f"Image file not found: {image_path}. Skipping this entry.")
                    skipped_missing_images += 1
                    continue # Skip this row if the H5 file doesn't exist

                target_value = row[self.target_column]
                if pd.isna(target_value):
                     logging.warning(f"NaN target value found for {image_filename} in {typhoon_id}. Skipping this entry.")
                     continue # Skip if target is NaN

                valid_image_paths.append(image_path)
                # Ensure target is float for regression loss functions
                valid_targets.append(float(target_value))

            # Generate sequences from the valid data for this typhoon
            if len(valid_image_paths) >= self.sequence_length:
                step_size = step_size  # 每隔3个时间步取一个序列（可调整）
                for i in range(0, len(valid_image_paths) - self.sequence_length + 1, step_size):
                    seq_paths = valid_image_paths[i : i + self.sequence_length]
                    seq_targets = valid_targets[i : i + self.sequence_length]
                    all_sequences.append((seq_paths, seq_targets))                
            else:
                skipped_short_typhoons += 1
                # logging.info(f"Typhoon {typhoon_id} has only {len(valid_image_paths)} valid images, less than sequence length {self.sequence_length}. Skipping.")


        if skipped_missing_images > 0:
             logging.warning(f"Skipped {skipped_missing_images} entries due to missing H5 image files or missing 'file_1' in CSV.")
        if skipped_short_typhoons > 0:
             logging.warning(f"Skipped {skipped_short_typhoons} typhoons because they had fewer valid images than the required sequence length.")

        return all_sequences

    def __len__(self) -> int:
        """Returns the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a single sequence of images and their corresponding targets.
        """
        image_paths, targets = self.sequences[idx]

        sequence_images = []
        for img_path in image_paths:
            try:
                with h5py.File(img_path, 'r') as f:
                    # Access the '/Infrared' dataset
                    image_data = f['Infrared'][:] # Load data into numpy array
                    
                    # Ensure data is float32 for PyTorch and normalization
                    image_data = image_data.astype(np.float32)

                    if self.normalize:
                         image_data = normalize_image(image_data, self.temp_min, self.temp_max)

                    # Add channel dimension (C, H, W) - Infrared is single channel
                    image_data = image_data[np.newaxis, :, :]
                    sequence_images.append(torch.from_numpy(image_data))

            except Exception as e:
                logging.error(f"Error loading or processing H5 file {img_path}: {e}")
                # Handle error: return None or raise exception, or return placeholder?
                # For simplicity, let's raise it here, but might need robust handling
                raise IOError(f"Could not load H5 file: {img_path}") from e

        # Stack images into a sequence tensor: (Sequence Length, C, H, W)
        images_tensor = torch.stack(sequence_images, dim=0)
        # Convert targets to a tensor: (Sequence Length)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        return images_tensor, targets_tensor

def get_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates train and validation DataLoaders based on the configuration.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A tuple containing (train_loader, val_loader, test_typhoon_ids).
        test_typhoon_ids are returned for potential later evaluation.
    """
    logging.info("Setting up DataLoaders...")
    # Load all typhoon IDs from metadata.json
    try:
        with open(config['METADATA_JSON_PATH'], 'r') as f:
            all_typhoon_metadata = json.load(f)
        all_typhoon_ids = list(all_typhoon_metadata.keys())
        logging.info(f"Loaded {len(all_typhoon_ids)} total typhoon IDs from {config['METADATA_JSON_PATH']}")
    except FileNotFoundError:
        logging.error(f"Metadata JSON file not found at {config['METADATA_JSON_PATH']}!")
        raise
    except json.JSONDecodeError:
         logging.error(f"Error decoding JSON from {config['METADATA_JSON_PATH']}!")
         raise

    if not all_typhoon_ids:
        raise ValueError("No typhoon IDs found in metadata.json.")

    # Split typhoon IDs into train, validation, and test sets
    # Ensure reproducibility with the fixed seed
    train_val_ids, test_ids = train_test_split(
        all_typhoon_ids,
        test_size=(1.0 - config['TRAIN_RATIO'] - config['VAL_RATIO']),
        random_state=config['SEED']
    )
    # Adjust validation split relative to the combined train+val set size
    relative_val_size = config['VAL_RATIO'] / (config['TRAIN_RATIO'] + config['VAL_RATIO'])
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=relative_val_size,
        random_state=config['SEED'] # Use same seed again for deterministic split within train/val
    )

    logging.info(f"Data split: {len(train_ids)} train IDs, {len(val_ids)} validation IDs, {len(test_ids)} test IDs.")

    if not train_ids:
         logging.warning("Warning: Training set is empty after split.")
    if not val_ids:
         logging.warning("Warning: Validation set is empty after split.")


    # Create Datasets
    train_dataset = TyphoonSequenceDataset(
        root_dir=config['ROOT_DIR'],
        typhoon_ids=train_ids,
        metadata_dir=config['METADATA_DIR'],
        image_dir=config['IMAGE_DIR'],
        sequence_length=config['SEQUENCE_LENGTH'],
        step_size=config["STEP_SIZE"],
        target_column=config['TARGET_COLUMN'],
        normalize=config['NORMALIZE'],
        temp_min=config['TEMP_MIN'],
        temp_max=config['TEMP_MAX']
    )

    val_dataset = TyphoonSequenceDataset(
        root_dir=config['ROOT_DIR'],
        typhoon_ids=val_ids,
        metadata_dir=config['METADATA_DIR'],
        image_dir=config['IMAGE_DIR'],
        sequence_length=config['SEQUENCE_LENGTH'],
        step_size=config["STEP_SIZE"],
        target_column=config['TARGET_COLUMN'],
        normalize=config['NORMALIZE'],
        temp_min=config['TEMP_MIN'],
        temp_max=config['TEMP_MAX']
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True, # Shuffle sequences for training
        num_workers=config['NUM_WORKERS'],
        pin_memory=True if config['DEVICE'] == torch.device('cuda') else False,
        drop_last=True, # Drop last incomplete batch if dataset size not divisible by batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False, # No need to shuffle validation data
        num_workers=config['NUM_WORKERS'],
        pin_memory=True if config['DEVICE'] == torch.device('cuda') else False,
        drop_last=False # Evaluate on all validation data
    )

    logging.info(f"Train DataLoader: {len(train_loader)} batches.")
    logging.info(f"Validation DataLoader: {len(val_loader)} batches.")

    # Return test IDs for potential use later (e.g., final evaluation script)
    return train_loader, val_loader, test_ids