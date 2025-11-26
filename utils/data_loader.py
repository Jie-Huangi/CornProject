import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .augmentation import CustomAugmentation
import config


class CornKernelDataset(Dataset):
    """Custom dataset for corn kernel classification"""

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or config.CLASS_TO_IDX

        self.images = []
        self.labels = []

        # Load all images and their labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '. png', '.bmp']:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def get_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4, image_size=224):
    """Create train and validation data loaders"""

    augmentation = CustomAugmentation()

    train_transform = augmentation.get_train_transform(image_size)
    val_transform = augmentation.get_val_transform(image_size)

    train_dataset = CornKernelDataset(train_dir, transform=train_transform)
    val_dataset = CornKernelDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset


def get_test_loader(test_dir, batch_size=32, num_workers=4, image_size=224):
    """Create test data loader"""

    augmentation = CustomAugmentation()
    test_transform = augmentation.get_test_transform(image_size)

    test_dataset = CornKernelDataset(test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader, test_dataset