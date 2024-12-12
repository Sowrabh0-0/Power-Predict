#This script is a basic version of the model i built, to run this please put it in the root folder of the project and run it
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_gpu_stats():
    """Log GPU memory stats to debug CUDA OOM issues."""
    if torch.cuda.is_available():
        logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Excel file containing refractive error data
excel_file_path = './ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/acuityvalues.xlsx'
acuity_data = pd.read_excel(excel_file_path)

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png']

# Function to load and organize IR images
def load_images_from_directory(directory_path):
    images_metadata = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                if "IR" in file:
                    file_path = os.path.join(root, file)
                    eye_type = "OD" if "RIGHT" in file else "OS"
                    images_metadata.append({
                        'filename': file,
                        'eye_type': eye_type,
                        'path': file_path
                    })
    return images_metadata

# Function to map images to acuity data
def map_images_to_acuity(images_metadata, acuity_data):
    mapped_data = []
    for meta in images_metadata:
        folder_name = os.path.basename(os.path.dirname(meta['path']))
        try:
            patient_id = int(folder_name)
        except ValueError:
            continue
        eye = meta['eye_type']
        patient_data = acuity_data[acuity_data['patient'] == patient_id]
        if not patient_data.empty:
            if eye == 'OD':
                meta['sphere'] = patient_data['r sphere'].values[0]
                meta['cylinder'] = patient_data['r cylinder'].values[0]
            else:
                meta['sphere'] = patient_data['l sphere'].values[0]
                meta['cylinder'] = patient_data['l cylinder'].values[0]
            mapped_data.append(meta)
    return mapped_data

# Validate data after mapping
def validate_data(mapped_images):
    valid_images = []
    valid_spheres = [float(meta['sphere']) for meta in mapped_images if pd.notna(meta['sphere']) and abs(meta['sphere']) != float('inf')]
    valid_cylinders = [float(meta['cylinder']) for meta in mapped_images if pd.notna(meta['cylinder']) and abs(meta['cylinder']) != float('inf')]

    mean_sphere = sum(valid_spheres) / len(valid_spheres) if valid_spheres else 0.0
    mean_cylinder = sum(valid_cylinders) / len(valid_cylinders) if valid_cylinders else 0.0

    for meta in mapped_images:
        try:
            sphere = float(meta['sphere']) if pd.notna(meta['sphere']) else mean_sphere
            cylinder = float(meta['cylinder']) if pd.notna(meta['cylinder']) else mean_cylinder
            meta['sphere'] = sphere
            meta['cylinder'] = cylinder
            valid_images.append(meta)
        except ValueError:
            continue

    return valid_images

# Custom dataset class
class RedReflexDataset(Dataset):
    def __init__(self, images_metadata, transform=None):
        self.images_metadata = images_metadata
        self.transform = transform

    def __len__(self):
        return len(self.images_metadata)

    def __getitem__(self, idx):
        img_path = self.images_metadata[idx]['path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sphere = self.images_metadata[idx].get('sphere', 0.0)
        cylinder = self.images_metadata[idx].get('cylinder', 0.0)
        return image, torch.tensor(sphere, dtype=torch.float32), torch.tensor(cylinder, dtype=torch.float32)

# Function to initialize the DataLoader
def get_dataloader(images_metadata, batch_size=64, num_workers=4):
    dataset = RedReflexDataset(images_metadata, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Improved Model with Residual and Depthwise Separable Convolutions
class ImprovedCustomModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedCustomModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Additional block
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),  # Adjust linear layer input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training Loop
def train_model(dataloader, model, criterion, optimizer, scheduler, epochs):
    scaler = torch.amp.GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, spheres, cylinders in dataloader:
            images, spheres, cylinders = images.to(device), spheres.to(device), cylinders.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss_sphere = criterion(outputs[:, 0], spheres)
                loss_cylinder = criterion(outputs[:, 1], cylinders)
                loss = loss_sphere + loss_cylinder

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()

            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            scheduler.step()

            running_loss += loss.item()

        logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Main block
if __name__ == "__main__":
    dataset_path = './ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/Images'
    images_metadata = load_images_from_directory(dataset_path)
    mapped_images = map_images_to_acuity(images_metadata, acuity_data)
    valid_mapped_images = validate_data(mapped_images)

    batch_size = 64
    epochs = 150

    dataloader = get_dataloader(valid_mapped_images, batch_size=batch_size, num_workers=4)
    model = ImprovedCustomModel(num_classes=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=epochs)

    train_model(dataloader, model, criterion, optimizer, scheduler, epochs)
    torch.save(model.state_dict(), "improved_model.pth")
    logging.info("Training complete. Model saved to improved_model.pth.")
