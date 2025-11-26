import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
from pathlib import Path
import json

import config
from models.squeezenet_ghost import squeezenet1_1
from utils.data_loader import get_data_loaders
from utils.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_DIR / 'training.log'),
            logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, device, config_dict):
        self.model = model.to(device)
        self.device = device
        self.config = config_dict

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )

        self.best_val_acc = 0
        self.metrics_calculator = MetricsCalculator(config.CLASS_NAMES)

    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)

        for batch_idx, (images, labels, _) in enumerate(pbar, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            avg_loss = total_loss / batch_idx
            acc_so_far = correct / total if total > 0 else 0.0

            # 更新 tqdm 进度条显示
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc_so_far:.4f}'
            })

        epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        """Validate on validation set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # 使用 tqdm 创建进度条
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=True)

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(pbar, 1):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_acc_so_far = correct / total if total > 0 else 0.0

                # 更新 tqdm 进度条显示
                pbar.set_postfix({
                    'acc': f'{val_acc_so_far:.4f}'
                })

        val_acc = correct / total if total > 0 else 0.0
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds)

        return val_acc, metrics, all_labels, all_preds

    def train(self, train_loader, val_loader, num_epochs):
        """Train the model"""
        logger.info(f"Starting training for {num_epochs} epochs")

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_metrics': []
        }

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate
            val_acc, metrics, all_labels, all_preds = self.validate(val_loader, epoch)
            history['val_acc'].append(val_acc)
            history['val_metrics'].append(metrics)

            logger.info(f"Train Loss: {train_loss:. 4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Acc: {val_acc:. 4f}, Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

            # Save checkpoint
            self._save_checkpoint(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

        # Save training history
        with open(config.LOG_DIR / 'training_history. json', 'w') as f:
            # Convert numpy values to float for JSON serialization
            history_serializable = {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'val_acc': [float(x) for x in history['val_acc']],
                'val_metrics': history['val_metrics']
            }
            json.dump(history_serializable, f, indent=4)

        logger.info("Training completed!")
        return history

    def _save_checkpoint(self, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }

        # Save last checkpoint
        torch.save(checkpoint, config.LAST_CHECKPOINT_PATH)

        # Save best checkpoint
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(checkpoint, config.CHECKPOINT_PATH)
            logger.info(f"Best model saved with accuracy: {val_acc:.4f}")


def main():
    logger.info("=" * 50)
    logger.info("Corn Kernel Classification Training")
    logger.info("=" * 50)

    # Device
    device = torch.device(config.TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = squeezenet1_1(
        num_classes=config.TRAIN_CONFIG['num_classes'],
        dropout=config.TRAIN_CONFIG['dropout']
    )
    logger.info("Model created: SqueezeNet with Ghost Module")

    # Data loaders
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        train_dir=config.DATA_CONFIG['train_path'],
        val_dir=config.DATA_CONFIG['val_path'],
        batch_size=config.DATA_CONFIG['batch_size'],
        num_workers=config.DATA_CONFIG['num_workers'],
        image_size=config.DATA_CONFIG['image_size']
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Trainer
    trainer = Trainer(model, device, config.TRAIN_CONFIG)

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.TRAIN_CONFIG['num_epochs']
    )


if __name__ == '__main__':
    main()