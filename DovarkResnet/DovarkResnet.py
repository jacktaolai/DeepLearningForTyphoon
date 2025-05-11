import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
import os
from pathlib import Path
from datetime import datetime
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import warnings
import json
from torch.nn.utils import clip_grad_norm_
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", category=UserWarning)

# --- Custom Modules (Inspired by Dvorak Analysis) ---
class ChannelAttention(nn.Module):
    """Simple Squeeze-and-Excitation Channel Attention"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class DvorakConv(nn.Module):
    """
    Dvorak-inspired Convolution Block integrating circular and bar-like receptive fields.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.circular = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bar = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, (3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels // 2, out_channels // 2, (1, 3), padding=(0, 1))
        )
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        circ_feat = self.circular(x)
        bar_feat = self.bar(x)
        combined_feat = torch.cat([circ_feat, bar_feat], dim=1)
        fused_feat = self.fusion(combined_feat)
        return self.relu(fused_feat)

# --- Model Class with DvorakConv Integration ---
class ResNetDvorak(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.dvorak_conv = DvorakConv(in_channels=512, out_channels=128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=128, out_features=num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dvorak_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        """Extract features from the dvorak_conv layer for Grad-CAM."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.dvorak_conv(x)
        return features

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.hook_handles = []
        self._register_hooks()
    
    def _save_gradients(self, grad):
        self.gradients = grad
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def generate(self, input_image, target_class=None):
        self.model.eval()
        input_image = input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        output[:, target_class].backward()
        
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.features, dim=1)
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().detach().numpy()
    
    def release_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# --- TyphoonClassifier Class ---
class TyphoonClassifier:
    def __init__(self, data_root, train_name=None, log_dir="runs", checkpoint_dir="checkpoints", heatmap_dir="heatmaps"):
        self.data_root = Path(data_root)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.heatmap_dir = Path(heatmap_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.heatmap_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        if train_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.train_name = f"train_{timestamp}"
        else:
            self.train_name = train_name
        
        self.standardize_range = (150, 350)
        self.downsample_size = (224, 224)
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.max_epochs = 72
        self.num_workers = 0 if os.name == 'nt' else 16
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience = 20
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.early_stop = False
        
        self.model = self._init_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.accuracy = Accuracy(task='multiclass', num_classes=7).to(self.device)
        
        self.train_loader, self.val_loader, self.test_loader, self.test_indices = self._init_dataloaders()
        self._init_tensorboard()
    
    def _init_tensorboard(self):
        tb_info_path = self.checkpoint_dir / f'tensorboard_info_{self.train_name}.json'
        if tb_info_path.exists():
            try:
                with open(tb_info_path, 'r') as f:
                    tb_info = json.load(f)
                self.tensorboard_run_dir = Path(tb_info['run_dir'])
                print(f"Resuming TensorBoard logging in existing directory: {self.tensorboard_run_dir}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode TensorBoard info file {tb_info_path}. Creating new log directory.")
                self.tensorboard_run_dir = self.log_dir / self.train_name / datetime.now().strftime("%Y%m%d_%H%M%S_corrupt_recover")
                self.tensorboard_run_dir.mkdir(parents=True, exist_ok=True)
                with open(tb_info_path, 'w') as f:
                    json.dump({'run_dir': str(self.tensorboard_run_dir)}, f)
                print(f"Created new TensorBoard directory: {self.tensorboard_run_dir}")
        else:
            self.tensorboard_run_dir = self.log_dir / self.train_name
            with open(tb_info_path, 'w') as f:
                json.dump({'run_dir': str(self.tensorboard_run_dir)}, f)
            print(f"Created new TensorBoard directory: {self.tensorboard_run_dir}")
        
        self.writer = SummaryWriter(self.tensorboard_run_dir)
    
    def _init_model(self):
        model = ResNetDvorak(num_classes=7)
        return model.to(self.device)
    
    def _transform_func(self, image_ray):
        image_ray = np.clip(image_ray, *self.standardize_range)
        image_ray = (image_ray - self.standardize_range[0]) / (self.standardize_range[1] - self.standardize_range[0])
        if image_ray.ndim == 3:
            image_ray = image_ray.squeeze(0)
        if self.downsample_size != (512, 512) and image_ray.shape != self.downsample_size:
            image_ray = torch.Tensor(image_ray)
            image_ray = torch.reshape(image_ray, [1, 1, *image_ray.size()])
            image_ray = nn.functional.interpolate(
                image_ray,
                size=self.downsample_size,
                mode='bilinear',
                align_corners=False
            )
            image_ray = torch.reshape(image_ray, [image_ray.size()[2], image_ray.size()[3]])
            image_ray = image_ray.numpy()
        return image_ray
    
    def _image_filter(self, image):
        return image.grade() < 7
    
    def _init_dataloaders(self):
        images_path = str(self.data_root / "image") + "/"
        metadata_path = str(self.data_root / "metadata") + "/"
        json_path = str(self.data_root / "metadata.json")
        
        dataset = DigitalTyphoonDataset(
            images_path,
            metadata_path,
            json_path,
            'grade',
            filter_func=self._image_filter,
            transform_func=self._transform_func,
            verbose=False
        )
        
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        if train_size == 0 or val_size == 0 or test_size == 0:
            raise ValueError(f"Dataset split resulted in zero size for one or more sets: train={train_size}, val={val_size}, test={test_size}. Total size: {dataset_size}. Check data_root and filters.")
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.seed)
        )
        self.test_indices = test_set.indices
        
        log_file = self.checkpoint_dir / f"dataloader_log_{self.train_name}.txt"
        with open(log_file, 'w') as f:
            f.write(f"=== DataLoader Configuration Log ===\n")
            f.write(f"Training Name: {self.train_name}\n")
            f.write(f"Log Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset total size: {dataset_size}\n")
            f.write(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}\n\n")
            f.write(f"train_indices ({len(train_set.indices)}):\n")
            f.write(str(train_set.indices)+"\n\n")
            f.write(f"val_indices ({len(val_set.indices)}):\n")
            f.write(str(val_set.indices)+"\n\n")
            f.write(f"test_indices ({len(test_set.indices)}):\n")
            f.write(str(test_set.indices)+"\n")
        
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader, self.test_indices
    
    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'current_epoch': self.current_epoch,
            'train_name': self.train_name,
            'test_indices': self.test_indices
        }
        
        checkpoint_path = self.checkpoint_dir / f'last_checkpoint_{self.train_name}.pth'
        torch.save(state, checkpoint_path)
        
        if is_best:
            for old_best in self.checkpoint_dir.glob(f'best_model_{self.train_name}_epoch*_acc*.pth'):
                old_best.unlink()
            best_checkpoint_path = self.checkpoint_dir / f'best_model_{self.train_name}_epoch{epoch+1}_acc{self.best_val_acc:.4f}.pth'
            torch.save(state, best_checkpoint_path)
            print(f"Saved best model to {best_checkpoint_path}")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self):
        checkpoint_path = self.checkpoint_dir / f'last_checkpoint_{self.train_name}.pth'
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except KeyError:
                    print("Warning: 'scheduler_state_dict' not found in checkpoint. Skipping.")
                except RuntimeError as e:
                    print(f"Warning: Could not load scheduler state_dict: {e}. Skipping.")
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                self.current_epoch = checkpoint.get('current_epoch', 0)
                self.test_indices = checkpoint.get('test_indices', self.test_indices)
                print(f"Loaded checkpoint from epoch {self.current_epoch}. Best val acc: {self.best_val_acc:.4f}")
                return True
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}")
                return False
        return False
    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        self.accuracy.reset()
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} (Train)', 
                 dynamic_ncols=True) as progress_bar:
            for images, labels in progress_bar:
                images = torch.Tensor(images).float().to(self.device, non_blocking=True)
                labels = torch.Tensor(labels).long().to(self.device, non_blocking=True)
                images = images.unsqueeze(1)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                    self.optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                self.accuracy.update(predicted, labels)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_train_loss = train_loss / len(self.train_loader)
        train_acc = self.accuracy.compute()
        
        return avg_train_loss, train_acc.item()
    
    def _save_heatmap(self, image, cam, label, index, mode='val'):
        """Save heatmap overlaid on the original image."""
        image_np = image.squeeze().cpu().detach().numpy()
        cam = np.uint8(255 * cam)
        cam = np.array(Image.fromarray(cam).resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR))
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_np, cmap='gray')
        plt.title(f'Original Image (Grade {label})')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(cam, cmap='jet', alpha=0.5, cbar=True)
        plt.imshow(image_np, cmap='gray', alpha=0.5)
        plt.title(f'Grad-CAM (Grade {label})')
        plt.axis('off')
        
        heatmap_path = self.heatmap_dir / f'{mode}_heatmap_grade_{label}_idx_{index}_epoch_{self.current_epoch+1}.png'
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {heatmap_path}")
    
    def validate(self, epoch, mode='val'):
        self.model.eval()
        loader = self.val_loader if mode == 'val' else self.test_loader
        total_loss = 0.0
        self.accuracy.reset()
        all_indices = []
        all_labels = []
        all_predicted = []
        all_probabilities = []
        desc = 'Validation' if mode == 'val' else 'Testing'
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(self.model, self.model.dvorak_conv)
        
        # Track one image per grade
        grade_examples = {i: None for i in range(7)}
        
        with torch.no_grad():
            with tqdm(loader, desc=f'Epoch {epoch+1}/{self.max_epochs} ({desc})',
                    dynamic_ncols=True, disable=mode=='test' and epoch > 0) as progress_bar:
                for batch_idx, (images, labels) in enumerate(progress_bar):
                    images = torch.Tensor(images).float().to(self.device, non_blocking=True)
                    labels = torch.Tensor(labels).long().to(self.device, non_blocking=True)
                    images = images.unsqueeze(1)
                    
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    self.accuracy.update(predicted, labels)
                    
                    start_idx_in_subset = batch_idx * self.batch_size
                    current_batch_subset_indices = range(start_idx_in_subset, min(start_idx_in_subset + len(images), len(loader.dataset)))
                    original_indices = [loader.dataset.indices[i] for i in current_batch_subset_indices]
                    
                    all_indices.extend(original_indices)
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Collect images for heatmap
                    if mode == 'val' and batch_idx == len(loader) - 1:
                        # For validation, only use the last batch
                        for i, label in enumerate(labels.cpu().numpy()):
                            if grade_examples[label] is None:
                                grade_examples[label] = (images[i:i+1], label, original_indices[i])
                    elif mode == 'test':
                        # For testing, collect from all batches until each grade has an example
                        for i, label in enumerate(labels.cpu().numpy()):
                            if grade_examples[label] is None:
                                grade_examples[label] = (images[i:i+1], label, original_indices[i])
        
        avg_loss = total_loss / len(loader)
        accuracy = self.accuracy.compute()
        
        # Save heatmaps for each grade
        for label in range(7):
            if grade_examples[label] is not None:
                image, lbl, idx = grade_examples[label]
                cam = grad_cam.generate(image)
                self._save_heatmap(image, cam, lbl, idx, mode)
            else:
                print(f"No image found for grade {label} in {mode} set.")
        
        grad_cam.release_hooks()
        self._save_predictions_to_file(all_indices, all_labels, all_predicted, all_probabilities, mode)
        
        return avg_loss, accuracy.item()
    
    def _save_predictions_to_file(self, indices, labels, predicted, probabilities, mode='test'):
        filename = self.checkpoint_dir / f"{mode}_predictions_{self.train_name}.csv"
        print(f"\nSaving {mode} predictions to {filename}")
        
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['dataset_index', 'true_label', 'predicted_label']
            header.extend([f'prob_class_{i}' for i in range(7)])
            csv_writer.writerow(header)
            
            for i in range(len(indices)):
                row = [indices[i], labels[i], predicted[i]]
                row.extend(probabilities[i].tolist())
                csv_writer.writerow(row)
        
        print(f"{mode.capitalize()} prediction saving complete.")
    
    def train(self):
        print(f"Starting training: {self.train_name}")
        print(f"TensorBoard logs at: {self.tensorboard_run_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Heatmaps saved to: {self.heatmap_dir}")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Using {type(self.model).__name__} model.")
        resume_training = self.load_checkpoint()
        start_epoch = self.current_epoch if resume_training else 0
        try:
            for epoch in range(start_epoch, self.max_epochs):
                if self.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}!")
                    break
                self.current_epoch = epoch
                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc = self.validate(epoch, 'val')
                self.scheduler.step(val_acc)
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                self.save_checkpoint(epoch, is_best=is_best)
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
                print(f"\nEpoch {epoch+1}/{self.max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"No Improve: {self.epochs_no_improve}/{self.patience}")
                if self.epochs_no_improve >= self.patience:
                    self.early_stop = True
            print("\nTraining finished. Running final test...")
            test_loss, test_acc = self.validate(self.current_epoch, 'test')
            print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")
            self.writer.add_scalar('Accuracy/test', test_acc, self.current_epoch)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint(self.current_epoch)
            print(f"Checkpoint saved at epoch {self.current_epoch}")
        finally:
            self.writer.close()
            print("TensorBoard writer closed.")

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    train_name = "ResNetDvorak_4-5_100epoch_2005_2025"
    classifier = TyphoonClassifier(
        data_root="/media/jacktao/Document/AU/",
        train_name=train_name
    )
    classifier.train()