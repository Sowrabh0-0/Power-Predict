import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import logging
from score_model import ModelScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTrainer:
    def __init__(self, dataloader, score_file='model_scores.json', models_dir='models'):
        self.dataloader = dataloader
        self.model = self.initialize_model()
        self.criterion, self.optimizer = self.get_loss_and_optimizer()
        self.scorer = ModelScorer(self.model, self.dataloader, self.criterion, score_file, models_dir)
        self.best_metrics = None  

    def initialize_model(self):
        from torchvision.models import EfficientNet_B0_Weights
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = True
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
        return model.to(device)

    def get_loss_and_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=1e-4)
        return criterion, optimizer

    def train(self, epochs=30):
        best_val_loss = float('inf')  
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            logging.info(f"Starting epoch {epoch + 1}/{epochs}...")
            for i, (images, spheres, cylinders) in enumerate(self.dataloader):
                images, spheres, cylinders = images.to(device), spheres.to(device).view(-1), cylinders.to(device).view(-1)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss_sphere = self.criterion(outputs[:, 0], spheres)
                loss_cylinder = self.criterion(outputs[:, 1], cylinders)
                loss = loss_sphere + loss_cylinder
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 10 == 0:
                    logging.info(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(self.dataloader)}], Loss: {running_loss / (i + 1):.4f}")

            metrics = self.scorer.validate()
            val_loss = metrics["validation_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metrics = metrics  

            scheduler.step()

        self.scorer.save_best_model(self.best_metrics)
