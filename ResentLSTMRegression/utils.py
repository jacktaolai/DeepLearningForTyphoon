# utils.py
import torch
import random
import numpy as np
import os
import shutil
import logging

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The following two lines are sometimes needed for full reproducibility with CUDA
        # but can impact performance. Use cautiously if exact bitwise reproducibility is critical.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str = "last_checkpoint.pth.tar", best_filename: str = "best_model.pth.tar"):
    """
    Saves model and training parameters checkpoint.

    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): If True, saves this checkpoint as the best model seen so far.
        checkpoint_dir (str): Directory where checkpoints will be saved.
        filename (str): Name of the file for the latest checkpoint.
        best_filename (str): Name of the file for the best checkpoint.
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.debug(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        logging.info(f"Best model checkpoint saved to {best_filepath} (Epoch {state.get('epoch', '?')}, {state.get('metric_name', 'Loss')}: {state.get('best_metric_val', '?'):.4f})")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: torch.device = torch.device('cpu')) -> dict:
    """
    Loads model and training parameters from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to load the state into. Defaults to None.
        device (torch.device): The device to load the checkpoint onto.

    Returns:
        dict: The dictionary loaded from the checkpoint file (contains epoch, best_metric_val, etc.). Returns empty dict if file not found.
    """
    if not os.path.isfile(checkpoint_path):
        logging.warning(f"Checkpoint file not found at '{checkpoint_path}'. Starting from scratch.")
        return {}

    logging.info(f"Loading checkpoint from '{checkpoint_path}'...")
    # Load checkpoint onto the specified device directly
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f"Loaded model state_dict from epoch {checkpoint.get('epoch', '?')}")

    if optimizer and 'optimizer' in checkpoint:
        try:
             optimizer.load_state_dict(checkpoint['optimizer'])
             logging.info("Loaded optimizer state_dict.")
             # Move optimizer state to the correct device (important if resuming on GPU)
             for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
             logging.warning(f"Could not load optimizer state: {e}. Optimizer will start from scratch.")

    else:
        if optimizer:
             logging.warning("Optimizer state not found in checkpoint or optimizer not provided. Optimizer will start from scratch.")


    return checkpoint # Return the loaded data (epoch, best_loss, etc.)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, checkpoint_dir: str = '.', run_name: str = 'experiment', metric_name: str = 'val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            checkpoint_dir (str): Directory to save the best model checkpoint.
            run_name (str): Name of the current run, used for checkpoint filename.
            metric_name (str): Name of the metric being monitored (e.g., 'val_loss').
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.inf
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.metric_name = metric_name
        self.best_filename = f"best_model_earlystop_{self.run_name}.pth.tar"
        self.best_model_state = None # Store the best model state in memory

    def __call__(self, metric_val: float, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """
        Call this method at the end of each validation epoch.

        Args:
            metric_val (float): The validation metric value for the current epoch.
            epoch (int): Current epoch number.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer being used.
        """
        # Assuming lower metric is better (e.g., loss)
        score = -metric_val # Convert to score where higher is better for comparison logic

        if self.best_score is None:
            self.best_score = score
            self.metric_min = metric_val
            self.save_best_model(metric_val, epoch, model, optimizer)
        elif score < self.best_score + self.delta: # Check if score hasn't improved enough
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best {self.metric_name}: {self.metric_min:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
                logging.warning("--- Early stopping triggered ---")
        else: # Score improved
            self.best_score = score
            self.metric_min = metric_val
            self.save_best_model(metric_val, epoch, model, optimizer)
            self.counter = 0 # Reset counter

    def save_best_model(self, metric_val: float, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Saves model checkpoint if validation metric improves."""
        if self.verbose:
            logging.info(f'{self.metric_name} improved ({self.metric_min:.6f} --> {metric_val:.6f}). Saving model...')

        # Save the best model state dictionary separately for easy loading later if needed
        self.best_model_state = model.state_dict()

        # Also save a full checkpoint file for the best state (optional but good practice)
        state = {
            'epoch': epoch + 1, # Save the next epoch number to start from
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric_val': self.metric_min,
            'metric_name': self.metric_name
        }
        filepath = os.path.join(self.checkpoint_dir, self.best_filename)
        torch.save(state, filepath)
        logging.info(f"Saved early stopping best model checkpoint to {filepath}")