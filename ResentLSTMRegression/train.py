# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from dataset import get_data_loaders
from model import ResNetLSTM
from utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(targets, predictions):
    """Calculate various regression metrics"""
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Calculate metrics for each sequence step
    metrics = {}
    for step in range(targets.shape[1]):
        step_targets = targets[:, step]
        step_preds = predictions[:, step]
        
        # Mean Absolute Error
        mae = np.mean(np.abs(step_targets - step_preds))
        # Mean Squared Error
        mse = np.mean((step_targets - step_preds)**2)
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        # R-squared
        ss_res = np.sum((step_targets - step_preds)**2)
        ss_tot = np.sum((step_targets - np.mean(step_targets))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        metrics[f'MAE/Step_{step}'] = mae
        metrics[f'MSE/Step_{step}'] = mse
        metrics[f'RMSE/Step_{step}'] = rmse
        metrics[f'R2/Step_{step}'] = r2
    
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    """Runs one epoch of training."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)
    
    all_targets = []
    all_preds = []
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
        # Store for metrics calculation
        all_targets.append(targets.detach())
        all_preds.append(outputs.detach())
    
    # Calculate metrics
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    metrics = calculate_metrics(all_targets, all_preds)
    
    # Log metrics to TensorBoard
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    for name, value in metrics.items():
        writer.add_scalar(f'Train/{name}', value, epoch)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1} [Train] Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    return avg_loss

def validate_one_epoch(model, loader, criterion, device, epoch, writer, save_predictions=False):
    """Runs one epoch of validation."""
    model.eval()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)
    
    all_targets = []
    all_preds = []
    sample_details = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Store for metrics calculation
            all_targets.append(targets)
            all_preds.append(outputs)
            
            # Store sample details for final epoch
            if save_predictions and epoch == config.NUM_EPOCHS - 1:
                for i in range(min(3, len(images))):  # Save first 3 samples
                    sample = {
                        'sample_id': batch_idx * loader.batch_size + i,
                        'targets': targets[i].cpu().numpy(),
                        'predictions': outputs[i].cpu().numpy()
                    }
                    sample_details.append(sample)
    
    # Calculate metrics
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    metrics = calculate_metrics(all_targets, all_preds)
    
    # Log metrics to TensorBoard
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Validation', avg_loss, epoch)
    for name, value in metrics.items():
        writer.add_scalar(f'Validation/{name}', value, epoch)
    
    # Save predictions if this is the final validation
    if save_predictions and epoch == config.NUM_EPOCHS - 1:
        save_validation_results(sample_details, metrics)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1} [Val]   Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    return avg_loss

def save_validation_results(sample_details, metrics):
    """Save validation results to CSV and text files."""
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(config.OUTPUT_DIR, 'validation_metrics.csv'), index=False)
    
    # Save sample predictions to CSV
    preds_data = []
    for sample in sample_details:
        for step in range(len(sample['targets'])):
            preds_data.append({
                'sample_id': sample['sample_id'],
                'step': step,
                'target': sample['targets'][step],
                'prediction': sample['predictions'][step]
            })
    preds_df = pd.DataFrame(preds_data)
    preds_df.to_csv(os.path.join(config.OUTPUT_DIR, 'validation_predictions.csv'), index=False)
    
    # Save detailed text summaries
    summary_file = os.path.join(config.OUTPUT_DIR, 'validation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Model: {config.RUN_NAME}\n")
        f.write(f"Validation Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write("\nSample Predictions:\n")
        for sample in sample_details[:3]:  # First 3 samples
            f.write(f"\nSample {sample['sample_id']}:\n")
            for step in reversed(range(len(sample['targets']))):  # Reverse to show steps in order
                f.write(f"step{step}\n")
                f.write(f"Target: {sample['targets'][step]:.2f}, Pred: {sample['predictions'][step]:.2f}\n")

def main():
    """Main training loop."""
    logging.info("--- Starting Training ---")
    set_seed(config.SEED)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Data Loaders
    try:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        train_loader, val_loader, _ = get_data_loaders(config_dict)
    except Exception as e:
        logging.error(f"Failed to create data loaders: {e}")
        return
    
    # Model setup
    model = ResNetLSTM(
        resnet_model_name=config.RESNET_MODEL,
        pretrained=config.PRETRAINED,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        output_dim=1
    ).to(config.DEVICE)
    
        # Add model graph to TensorBoard
    try:
        # Get a sample batch from the training loader
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(config.DEVICE)
        
        # Add graph to TensorBoard
        writer.add_graph(model, sample_images)
        logging.info("Successfully added model graph to TensorBoard")
    except Exception as e:
        logging.warning(f"Failed to add model graph to TensorBoard: {e}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Checkpointing
    start_epoch = 0
    best_metric_val = float('inf')
    last_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth.tar")
    if os.path.exists(last_checkpoint_path):
        try:
            checkpoint = load_checkpoint(last_checkpoint_path, model, optimizer, config.DEVICE)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0)
                best_metric_val = checkpoint.get('best_metric_val', float('inf'))
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    
    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        delta=config.EARLY_STOPPING_DELTA,
        checkpoint_dir=config.CHECKPOINT_DIR,
        run_name=config.RUN_NAME,
        metric_name=config.BEST_CHECKPOINT_METRIC
    )
    
    # Training Loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, epoch, writer)
        
        # For final epoch, save predictions
        save_preds = (epoch == config.NUM_EPOCHS - 1)
        val_loss = validate_one_epoch(model, val_loader, criterion, config.DEVICE, epoch, writer, save_preds)
        
        # Checkpointing
        is_best = val_loss < best_metric_val
        if is_best:
            best_metric_val = val_loss
        
        if (epoch + 1) % config.SAVE_CHECKPOINT_EPOCH_FREQ == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric_val': best_metric_val,
                'metric_name': config.BEST_CHECKPOINT_METRIC
            }, is_best, config.CHECKPOINT_DIR, filename="last_checkpoint.pth.tar")
        
        # Early stopping
        early_stopper(val_loss, epoch, model, optimizer)
        if early_stopper.early_stop:
            break
    
    writer.close()
    logging.info("--- Training Finished ---")
    logging.info(f"Output files saved in: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()