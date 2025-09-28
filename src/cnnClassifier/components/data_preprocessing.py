import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from cnnClassifier.entity.config_entity import DataPreprocessingConfig

class TrashNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def _get_transforms(self, train=True):
        if train and self.config.augmentation:
            return transforms.Compose([
                transforms.Resize((self.config.image_size[0], self.config.image_size[1])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.config.image_size[0], self.config.image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def get_dataloaders(self):
        dataset = TrashNetDataset(
            data_dir=self.config.data_dir,
            transform=None
        )
        
        dataset_size = len(dataset)
        test_size = int(self.config.test_split * dataset_size)
        val_size = int(self.config.val_split * dataset_size)
        train_size = dataset_size - val_size - test_size

        if self.config.shuffle:
            torch.manual_seed(self.config.random_seed)

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Apply transforms
        train_dataset.dataset.transform = self._get_transforms(train=True)
        val_dataset.dataset.transform = self._get_transforms(train=False)
        test_dataset.dataset.transform = self._get_transforms(train=False)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
