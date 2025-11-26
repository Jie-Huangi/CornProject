import torch
import torch.nn.functional as F
from PIL import Image
import logging
from pathlib import Path
import cv2
import numpy as np

import config
from models.squeezenet_ghost import squeezenet1_1
from utils.augmentation import CustomAugmentation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.augmentation = CustomAugmentation()

        # Create model
        self.model = squeezenet1_1(
            num_classes=config.TRAIN_CONFIG['num_classes'],
            dropout=config.TRAIN_CONFIG['dropout']
        )

        # Load checkpoint
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path, return_probs=True):
        """Predict class for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = self.augmentation.get_test_transform(config.DATA_CONFIG['image_size'])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        class_name = config.IDX_TO_CLASS[pred_class]

        if return_probs:
            all_probs = {config.IDX_TO_CLASS[i]: probs[0, i].item()
                         for i in range(len(config.CLASS_NAMES))}
            return class_name, confidence, all_probs
        else:
            return class_name, confidence

    def predict_batch(self, image_dir, save_results=True):
        """Predict for all images in a directory"""
        image_dir = Path(image_dir)
        results = []

        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        logger.info(f"Found {len(image_files)} images to predict")

        for img_path in image_files:
            try:
                class_name, confidence, all_probs = self.predict_image(img_path)
                result = {
                    'image': img_path.name,
                    'predicted_class': class_name,
                    'confidence': confidence,
                    'all_probabilities': all_probs
                }
                results.append(result)
                logger.info(f"{img_path.name}: {class_name} ({confidence:.4f})")
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {str(e)}")

        if save_results:
            import json
            results_path = config.LOG_DIR / 'predictions.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to {results_path}")

        return results

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction on image"""
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        class_name, confidence, all_probs = self.predict_image(image_path)

        # Add text to image
        cv2.putText(image_np, f"{class_name} ({confidence:.2%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i, (class_n, prob) in enumerate(all_probs.items()):
            cv2.putText(image_np, f"{class_n}: {prob:.4f}", (10, 60 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            logger.info(f"Visualization saved to {save_path}")

        return image_np


def main():
    logger.info("=" * 50)
    logger.info("Corn Kernel Classification Prediction")
    logger.info("=" * 50)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create predictor
    predictor = Predictor(
        checkpoint_path=config.CHECKPOINT_PATH,
        device=device
    )

    # Example: Predict on a single image
    test_image = config.DATA_CONFIG['test_path'] / 'intact' / 'sample.jpg'
    if test_image.exists():
        class_name, confidence, all_probs = predictor.predict_image(test_image)
        logger.info(f"Prediction: {class_name} (Confidence: {confidence:.4f})")
        logger.info(f"All probabilities: {all_probs}")

    # Example: Predict on batch
    # results = predictor.predict_batch(config.DATA_CONFIG['test_path'])


if __name__ == '__main__':
    main()