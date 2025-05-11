import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import warnings
import json
from torch.nn.utils import clip_grad_norm_

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", category=UserWarning)

class TyphoonClassifier:
    def __init__(self, data_root, train_name=None, log_dir="runs", checkpoint_dir="checkpoints"):
        """
        初始化台风分类器
        
        参数:
            data_root: 数据根目录
            train_name: 训练名称，用于区分不同训练任务
            log_dir: TensorBoard日志目录
            checkpoint_dir: 检查点保存目录
        """
        # 初始化参数
        self.data_root = Path(data_root)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 添加随机种子控制
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # 设置训练名称
        if train_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.train_name = f"train_{timestamp}"
        else:
            self.train_name = train_name
        
        # 图像处理参数
        self.standardize_range = (150, 350)
        self.downsample_size = (224, 224)
        
        # 训练参数
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.max_epochs = 100
        self.num_workers = 0 if os.name == 'nt' else 16  # Windows下设为0
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.tensorboard_run_dir = None  # 用于保存TensorBoard运行目录

        # 初始化早停相关变量
        self.patience = 20  # 允许验证集性能不提升的连续epoch数
        self.best_val_acc = 0.0  # 记录最佳验证集准确率
        self.epochs_no_improve = 0  # 记录未提升的epoch数
        self.early_stop = False  # 是否触发早停
        
        # 初始化模型和工具
        self.model = self._init_model()
        self.criterion = nn.CrossEntropyLoss()
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.accuracy = Accuracy(task='multiclass', num_classes=7).to(self.device)
        
        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._init_dataloaders()
        
        # 初始化TensorBoard
        self._init_tensorboard()
        
    def _init_tensorboard(self):
        """初始化TensorBoard，考虑恢复训练的情况"""
        # 检查是否有保存的TensorBoard运行信息
        tb_info_path = self.checkpoint_dir / f'tensorboard_info_{self.train_name}.json'
        if tb_info_path.exists():
            with open(tb_info_path, 'r') as f:
                tb_info = json.load(f)
            self.tensorboard_run_dir = Path(tb_info['run_dir'])
            print(f"Resuming TensorBoard logging in existing directory: {self.tensorboard_run_dir}")
        else:
            # 创建新的TensorBoard运行目录
            self.tensorboard_run_dir = self.log_dir / self.train_name
            # 保存TensorBoard运行信息
            with open(tb_info_path, 'w') as f:
                json.dump({'run_dir': str(self.tensorboard_run_dir)}, f)
            print(f"Created new TensorBoard directory: {self.tensorboard_run_dir}")
        
        # 初始化SummaryWriter
        self.writer = SummaryWriter(self.tensorboard_run_dir)
    
    def _init_model(self):
        # """初始化ConvNeXt模型"""
        # from torchvision.models import convnext_tiny
        # model = convnext_tiny(weights=None)
        # # 修改第一层卷积适应单通道输入
        # model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        # # 修改分类头
        # model.classifier[2] = nn.Linear(768, 7)
        # return model.to(self.device)

        # """初始化ResNet18模型"""
        # model = models.resnet18(weights=None)
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), 
        #                        padding=(3, 3), bias=False)
        # model.fc = nn.Linear(in_features=512, out_features=7, bias=True)
        # return model.to(self.device)

        """初始化Vision Transformer模型"""
        from torchvision.models import vit_b_16
        model = vit_b_16(weights=None)
        # 修改patch嵌入层适应单通道输入
        model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        # 修改分类头
        model.heads.head = nn.Linear(768, 7)
        return model.to(self.device)
    
    def _transform_func(self, image_ray):
        """图像预处理函数"""
        image_ray = np.clip(image_ray, *self.standardize_range)
        image_ray = (image_ray - self.standardize_range[0]) / (self.standardize_range[1] - self.standardize_range[0])
        
        if self.downsample_size != (512, 512):
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
        """图像过滤函数"""
        return image.grade() < 7
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        # 数据路径
        images_path = str(self.data_root / "image") + "/"
        metadata_path = str(self.data_root / "metadata") + "/"
        json_path = str(self.data_root / "metadata.json")
        
        # 创建数据集
        dataset = DigitalTyphoonDataset(
            images_path,
            metadata_path,
            json_path,
            'grade',
            filter_func=self._image_filter,
            transform_func=self._transform_func,
            verbose=False
        )
        
        # 分割数据集
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        #记录Dataset分组情况信息到文件
        log_file = self.checkpoint_dir / f"dataloader_log_{self.train_name}.txt"
        with open(log_file, 'a') as f:
            # 写入头部信息
            f.write(f"=== DataLoader Configuration Log ===\n")
            f.write(f"Training Name: {self.train_name}\n")
            f.write(f"Log Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"train\n\n")
            f.write(str(train_set.indices)+"\n")
            f.write(f"val\n\n")
            f.write(str(val_set.indices)+"\n")
            f.write(f"test\n\n")
            f.write(str(test_set.indices)+"\n")

        # 创建DataLoader并启用预取
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
        
        return train_loader, val_loader, test_loader
    
    def _log_images(self, images, labels, predicted, epoch, mode='train'):
        """记录图像到TensorBoard"""
        # 只记录前8个图像
        images = images[:8].cpu()
        labels = labels[:8].cpu()
        predicted = predicted[:8].cpu()
        
        # 反标准化图像
        images = images * (self.standardize_range[1] - self.standardize_range[0]) + self.standardize_range[0]
        
        # 创建网格图像
        grid = make_grid(images.unsqueeze(1), normalize=True, scale_each=True)
        
        # 添加标题
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Labels: {labels.tolist()}\nPredicted: {predicted.tolist()}")
        plt.axis('off')
        
        # 记录到TensorBoard
        self.writer.add_figure(f'{mode}_images', fig, epoch)
        plt.close(fig)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存训练状态"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'current_epoch': self.current_epoch,
            'train_name': self.train_name  # 保存训练名称
        }
        
        # 保存常规checkpoint
        checkpoint_path = self.checkpoint_dir / f'last_checkpoint_{self.train_name}.pth'
        torch.save(state, checkpoint_path)
        
        # 如果是当前最佳模型，单独保存
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / f'best_model_{self.train_name}_epoch{epoch+1}_acc{self.best_val_acc:.4f}.pth'
            torch.save(state, best_checkpoint_path)
    
    def load_checkpoint(self):
        """加载训练状态"""
        checkpoint_path = self.checkpoint_dir / f'last_checkpoint_{self.train_name}.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            self.current_epoch = checkpoint['current_epoch']
            print(f"Loaded checkpoint from epoch {self.current_epoch}. Best val acc: {self.best_val_acc:.4f}")
            return True
        return False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # 使用tqdm的替代方式避免冲突
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs}', 
                 dynamic_ncols=True) as progress_bar:
            for images, labels in progress_bar:
                # 数据预处理
                images = torch.Tensor(images).float().to(self.device, non_blocking=True)
                labels = torch.Tensor(labels).long().to(self.device, non_blocking=True)
                images = images.unsqueeze(1)  # 添加通道维度
                
                # 更新后的混合精度API
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    self.optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 梯度裁剪（关键步骤！）
                clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                # 计算指标
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # 记录第一批图像
                if progress_bar.n == 0:
                    self._log_images(images.squeeze(1), labels, predicted, epoch, 'train')
        
        # 计算平均指标
        avg_train_loss = train_loss / len(self.train_loader)
        train_acc = train_correct / train_total
        
        return avg_train_loss, train_acc
    
    def validate(self, epoch, mode='val'):
        """验证或测试"""
        self.model.eval()
        loader = self.val_loader if mode == 'val' else self.test_loader
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = torch.Tensor(images).float().to(self.device, non_blocking=True)
                labels = torch.Tensor(labels).long().to(self.device, non_blocking=True)
                images = images.unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # 记录第一批图像
                if total_samples == labels.size(0):  # 只记录第一批
                    self._log_images(images.squeeze(1), labels, predicted, epoch, mode)
        
        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """训练主循环（含早停机制）"""
        print(f"Starting training: {self.train_name}")
        print(f"TensorBoard logs at: {self.tensorboard_run_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        

        
        # 尝试加载之前的checkpoint
        resume_training = self.load_checkpoint()
        start_epoch = self.current_epoch if resume_training else 0
        
        try:
            for epoch in range(start_epoch, self.max_epochs):
                if self.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}!")
                    break
                
                self.current_epoch = epoch
                
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_acc = self.validate(epoch, 'val')
                
                # 学习率调整
                self.scheduler.step(val_acc)
                
                # 更新最佳准确率并检查早停条件
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0  # 重置计数器
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_no_improve += 1
                    self.save_checkpoint(epoch)
                    
                    # 检查是否触发早停
                    if self.epochs_no_improve >= self.patience:
                        self.early_stop = True
                
                # 记录到TensorBoard
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                print(f"\nEpoch {epoch+1}/{self.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"No Improve: {self.epochs_no_improve}/{self.patience}")
            
            # 最终测试（无论是否早停都执行）
            test_loss, test_acc = self.validate(0, 'test')
            print(f"\nFinal Test Accuracy: {test_acc:.4f}")
            self.writer.add_scalar('Accuracy/test', test_acc, self.current_epoch)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint(self.current_epoch)
            print(f"Checkpoint saved at epoch {self.current_epoch}")
            
        finally:
            self.writer.close()

if __name__ == '__main__':
    # 设置环境变量解决OpenMP问题
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 定义训练名称，可以包含模型类型、epoch数、数据量等信息
    train_name = "visionTransform_4-5_100epoch_2005_2025"
    
    classifier = TyphoonClassifier(
        data_root="/media/jacktao/Document/AU/",
        train_name=train_name
    )
    print(f"Training name: {classifier.train_name}")
    print(f"Using device: {classifier.device}")
    print(f"Batch size: {classifier.batch_size}")
    print(f"Max epochs: {classifier.max_epochs}")
    classifier.train()