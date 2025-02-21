import torch
import numpy as np
from torch import nn, optim
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights
from utils.resize_pad import ResizeAndPad

class MushroomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/Mushrooms", batch_size=32, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.class_weights = None
        self.transform = transforms.Compose([
            ResizeAndPad((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.3913, 0.3695, 0.2812], std=[0.2497, 0.2267, 0.2234]),
        ])

    def setup(self, stage=None):
        full_dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Calculate class weights
        class_sample_counts = torch.tensor([len(np.where(full_dataset.targets == t)[0]) for t in np.unique(full_dataset.targets)])
        self.class_weights = 1. / class_sample_counts.float()
        self.class_weights = self.class_weights / self.class_weights.sum() 
        self.class_weights = self.class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=10,persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=10,persistent_workers=True)

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.mean((preds == labels).float())
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'preds': preds, 'labels': labels}
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0]
