import torch
import logging
from pathlib import Path
import json

import config
from models.squeezenet_ghost import squeezenet1_1
from utils.data_loader import get_test_loader
from utils.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, device, checkpoint_path):
        self.model = model.to(device)
        self.device = device
        self.metrics_calculator = MetricsCalculator(config.CLASS_NAMES)

        # Load checkpoint
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for images, labels, paths in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_paths.extend(paths)

        test_acc = correct / total
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds)

        return test_acc, metrics, all_labels, all_preds, all_paths

    def generate_report(self, test_acc, metrics, all_labels, all_preds):
        """Generate evaluation report"""
        logger.info("\n" + "=" * 50)
        logger.info("Test Results")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {test_acc:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(self.metrics_calculator.get_classification_report(all_labels, all_preds))

        # Save confusion matrix
        cm_path = config.LOG_DIR / 'confusion_matrix.png'
        self.metrics_calculator.plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Save metrics plot
        metrics_path = config.LOG_DIR / 'metrics.png'
        self.metrics_calculator.plot_metrics(metrics, save_path=metrics_path)
        logger.info(f"Metrics plot saved to {metrics_path}")


def main():
    logger.info("=" * 50)
    logger.info("Evaluating Model on Test Set")
    logger.info("=" * 50)

    # Device
    device = torch.device(config.TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = squeezenet1_1(
        num_classes=config.TRAIN_CONFIG['num_classes'],
        dropout=config.TRAIN_CONFIG['dropout']
    )

    # Evaluator
    evaluator = Evaluator(model, device, config.CHECKPOINT_PATH)

    # Test loader
    test_loader, test_dataset = get_test_loader(
        test_dir=config.DATA_CONFIG['test_path'],
        batch_size=config.DATA_CONFIG['batch_size'],
        num_workers=config.DATA_CONFIG['num_workers'],
        image_size=config.DATA_CONFIG['image_size']
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Evaluate
    test_acc, metrics, all_labels, all_preds, all_paths = evaluator.evaluate(test_loader)

    # Generate report
    evaluator.generate_report(test_acc, metrics, all_labels, all_preds)


if __name__ == '__main__':
    main()