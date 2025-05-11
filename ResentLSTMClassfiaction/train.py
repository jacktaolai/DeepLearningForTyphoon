# train.py (modified for classification)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import config
from dataset import get_data_loaders
from model import ResNetLSTM
from utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(targets, predictions, num_classes):
    """Calculate classification metrics"""
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # For multi-class classification
    if num_classes > 2:
        average_type = 'macro'
    else:
        average_type = 'binary'
    
    # 展平所有时间步
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, num_classes)
        predictions = torch.argmax(predictions, dim=1)
    
    targets = targets.flatten()
    predictions = predictions.flatten()

    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average_type)
    recall = recall_score(targets, predictions, average=average_type)
    f1 = f1_score(targets, predictions, average=average_type)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Add per-class metrics for multi-class
    if num_classes > 2:
        for class_idx in range(num_classes):
            class_precision = precision_score(targets, predictions, average=None)[class_idx]
            class_recall = recall_score(targets, predictions, average=None)[class_idx]
            class_f1 = f1_score(targets, predictions, average=None)[class_idx]
            
            metrics[f'precision_class_{class_idx}'] = class_precision
            metrics[f'recall_class_{class_idx}'] = class_recall
            metrics[f'f1_class_{class_idx}'] = class_f1
    
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, num_classes):
    """Runs one epoch of training for classification."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    all_targets = []
    all_preds = []
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.long()
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_targets.append(targets.detach())
        all_preds.append(preds.detach())
        
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate metrics
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    metrics = calculate_metrics(all_targets, all_preds, num_classes)
    
    # Log metrics
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    for name, value in metrics.items():
        writer.add_scalar(f'Train/{name}', value, epoch)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1} [Train] Loss: {avg_loss:.4f}, Acc: {metrics['accuracy']:.4f}, Time: {elapsed_time:.2f}s")
    return avg_loss

def validate_one_epoch(model, loader, criterion, device, epoch, writer, num_classes, save_predictions=False):
    """Runs one epoch of validation for classification."""
    model.eval()
    total_loss = 0.0
    start_time = time.time()
    
    all_targets = []
    all_preds = []
    all_probs = []
    sample_details = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.long()
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_targets.append(targets)
            all_preds.append(preds)
            all_probs.append(probs)
            
            if save_predictions and epoch == config.NUM_EPOCHS - 1:
                for i in range(min(3, len(images))):
                    sample = {
                        'sample_id': batch_idx * loader.batch_size + i,
                        'target': targets[i].item(),
                        'prediction': preds[i].item(),
                        'probabilities': probs[i].cpu().numpy()
                    }
                    sample_details.append(sample)
    
    # Calculate metrics
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    metrics = calculate_metrics(all_targets, all_preds, num_classes)
    
    # Log metrics
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Validation', avg_loss, epoch)
    for name, value in metrics.items():
        writer.add_scalar(f'Validation/{name}', value, epoch)
    
    # Save predictions if final epoch
    if save_predictions and epoch == config.NUM_EPOCHS - 1:
        save_classification_results(sample_details, metrics, num_classes)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1} [Val] Loss: {avg_loss:.4f}, Acc: {metrics['accuracy']:.4f}, Time: {elapsed_time:.2f}s")
    return avg_loss

def save_classification_results(sample_details, metrics, num_classes):
    """Save classification results to files."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(config.OUTPUT_DIR, 'classification_metrics.csv'), index=False)
    
    # Save predictions
    preds_data = []
    for sample in sample_details:
        preds_data.append({
            'sample_id': sample['sample_id'],
            'target': sample['target'],
            'prediction': sample['prediction'],
            **{f'prob_class_{i}': sample['probabilities'][i] for i in range(num_classes)}
        })
    preds_df = pd.DataFrame(preds_data)
    preds_df.to_csv(os.path.join(config.OUTPUT_DIR, 'classification_predictions.csv'), index=False)
    
    # Save summary
    summary_file = os.path.join(config.OUTPUT_DIR, 'classification_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Model: {config.RUN_NAME}\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        f.write("Validation Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write("\nSample Predictions:\n")
        for sample in sample_details[:3]:
            f.write(f"\nSample {sample['sample_id']}:\n")
            f.write(f"Target: {sample['target']}\n")
            f.write(f"Prediction: {sample['prediction']}\n")
            f.write("Probabilities:\n")
            for i, prob in enumerate(sample['probabilities']):
                f.write(f"Class {i}: {prob:.4f}\n")

def main():
    """Main training loop for classification."""
    logging.info("--- Starting Classification Training ---")
    set_seed(config.SEED)
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Data Loaders
    try:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        train_loader, val_loader, _ = get_data_loaders(config_dict)
    except Exception as e:
        logging.error(f"Failed to create data loaders: {e}")
        return
    
    # Model setup - output_dim should equal num_classes
    model = ResNetLSTM(
        resnet_model_name=config.RESNET_MODEL,
        pretrained=config.PRETRAINED,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        output_dim=config.NUM_CLASSES  # Important change for classification
    ).to(config.DEVICE)
    
    # Add graph to TensorBoard
    try:
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(config.DEVICE)
        writer.add_graph(model, sample_images)
    except Exception as e:
        logging.warning(f"Failed to add model graph: {e}")
    
    # Loss function - CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Checkpointing and early stopping
    start_epoch = 0
    best_metric_val = float('inf')
    last_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth.tar")
    if os.path.exists(last_checkpoint_path):
        try:
            checkpoint = load_checkpoint(last_checkpoint_path, model, optimizer, config.DEVICE)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                best_metric_val = checkpoint['best_metric_val']
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
    
    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, epoch, writer, config.NUM_CLASSES
        )
        
        save_preds = (epoch == config.NUM_EPOCHS - 1)
        val_loss = validate_one_epoch(
            model, val_loader, criterion, 
            config.DEVICE, epoch, writer, 
            config.NUM_CLASSES, save_preds
        )
        
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
                'metric_name': config.BEST_CHECKPOINT_METRIC,
                'num_classes': config.NUM_CLASSES
            }, is_best, config.CHECKPOINT_DIR, filename="last_checkpoint.pth.tar")
        
        if early_stopper.early_stop:
            break
    
    writer.close()
    logging.info("--- Training Finished ---")
    logging.info(f"Results saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()