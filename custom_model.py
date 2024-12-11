import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, input_size=(224, 224)):
        super(CustomCNN, self).__init__()
        
        # Dynamic feature extraction to calculate correct input size for linear layer
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        def pool2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calculate feature map size dynamically
        h, w = input_size
        
        # Convolution and pooling calculations
        h = pool2d_size_out(pool2d_size_out(pool2d_size_out(h)))
        w = pool2d_size_out(pool2d_size_out(pool2d_size_out(w)))

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dynamically calculate flattened feature size
        self.feature_size = 128 * h * w

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ModelTrainer:
    def __init__(self, dataloader, score_file='custom_model_scores.json', models_dir='custom_models'):
        self.dataloader = dataloader
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Determine input size from first batch
        sample_batch = next(iter(dataloader))
        sample_image = sample_batch[0][0]  # First image in the first batch
        input_size = (sample_image.shape[1], sample_image.shape[2])
        
        self.model = self.initialize_model(input_size)
        self.criterion, self.optimizer = self.get_loss_and_optimizer()
        self.best_metrics = None

    def initialize_model(self, input_size):
        model = CustomCNN(input_size=input_size)
        return model.to(device)

    def get_loss_and_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=5e-4)
        return criterion, optimizer

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, spheres, cylinders in self.dataloader:
                images = images.to(device)
                spheres = spheres.to(device).view(-1)
                cylinders = cylinders.to(device).view(-1)

                outputs = self.model(images)
                loss_sphere = self.criterion(outputs[:, 0], spheres)
                loss_cylinder = self.criterion(outputs[:, 1], cylinders)
                loss = loss_sphere + loss_cylinder
                
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        return {
            "validation_loss": avg_loss,
            "model_path": self.save_model(avg_loss)
        }

    def save_model(self, loss):
        model_filename = f'custom_model_loss_{loss:.4f}.pth'
        model_path = os.path.join(self.models_dir, model_filename)
        torch.save(self.model.state_dict(), model_path)
        return model_path

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

            # Validation
            metrics = self.validate()
            val_loss = metrics["validation_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metrics = metrics

            scheduler.step()

        # Save the best model
        if self.best_metrics:
            logging.info(f"Best model saved with loss: {best_val_loss:.4f}")
            logging.info(f"Model path: {self.best_metrics['model_path']}")