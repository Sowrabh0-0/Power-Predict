import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import logging

# Setup logging (Remove debug-level logging for data processing)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transformations (e.g., for EfficientNet)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
    transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),  # More aggressive cropping
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the Excel file containing refractive error data
excel_file_path = './Choithram Netralaya Data/acuityvalues.xlsx'
acuity_data = pd.read_excel(excel_file_path)

# Supported image formats (only jpg and png)
image_extensions = ['.jpg', '.jpeg', '.png']

# Function to load and organize IR images from the directory
def load_images_from_directory(directory_path):
    images_metadata = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                if "IR" in file:
                    file_path = os.path.join(root, file)
                    eye_type = "OD" if "RIGHT" in file else "OS"
                    date_info = file.split('_')[0]
                    images_metadata.append({
                        'filename': file,
                        'eye_type': eye_type,
                        'date_info': date_info,
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

# Validate data after mapping images to acuity data
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

            if abs(sphere) == float('inf') or abs(cylinder) == float('inf'):
                continue
            else:
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
        sphere = torch.tensor(float(self.images_metadata[idx]['sphere']), dtype=torch.float32)
        cylinder = torch.tensor(float(self.images_metadata[idx]['cylinder']), dtype=torch.float32)
        return image, sphere, cylinder

# Dataset and DataLoader initialization
def get_dataloader(images_metadata, batch_size=32, num_workers=6):
    dataset = RedReflexDataset(images_metadata, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    return dataloader

# Load EfficientNet model
def initialize_model():
    logging.info("Initializing EfficientNet model...")
    from torchvision.models import EfficientNet_B0_Weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
    return model.to(device)

# Loss function and optimizer
def get_loss_and_optimizer(model):
    criterion = nn.MSELoss()  # More sensitive to errors than SmoothL1Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    return criterion, optimizer

# Training loop with early stopping (save only the best model)
def train_model(dataloader, model, criterion, optimizer, epochs=30, early_stopping_patience=7):
    best_val_loss = float('inf')
    patience_counter = 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_path = 'models/best_efficientnet.pth'

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        logging.info(f"Starting epoch {epoch + 1}...")
        for i, (images, spheres, cylinders) in enumerate(dataloader):
            images, spheres, cylinders = images.to(device), spheres.to(device).view(-1), cylinders.to(device).view(-1)
            optimizer.zero_grad()
            outputs = model(images)
            predicted_spheres, predicted_cylinders = outputs[:, 0], outputs[:, 1]
            loss_sphere = criterion(predicted_spheres, spheres)
            loss_cylinder = criterion(predicted_cylinders, cylinders)
            loss = loss_sphere + loss_cylinder
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / (i + 1):.4f}")

        val_loss, _ = validate_model(dataloader, model, criterion)
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with Validation Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss for {patience_counter} epochs.")

        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered. Stopping training after {epoch + 1} epochs.")
            break

    logging.info("Training complete! Best validation loss: {:.4f}".format(best_val_loss))

# Validation function with MAE
# Validation function with MAE
def validate_model(dataloader, model, criterion):
    model.eval()
    running_val_loss = 0.0
    total_sphere_error = 0.0
    total_cylinder_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, spheres, cylinders in dataloader:
            images, spheres, cylinders = images.to(device), spheres.to(device).view(-1), cylinders.to(device).view(-1)

            # Forward pass
            outputs = model(images)
            predicted_spheres, predicted_cylinders = outputs[:, 0], outputs[:, 1]

            # Calculate losses
            loss_sphere = criterion(predicted_spheres, spheres)
            loss_cylinder = criterion(predicted_cylinders, cylinders)
            loss = loss_sphere + loss_cylinder

            running_val_loss += loss.item()

            # Track the error for accuracy calculation
            total_sphere_error += torch.sum(torch.abs(predicted_spheres - spheres)).item()
            total_cylinder_error += torch.sum(torch.abs(predicted_cylinders - cylinders)).item()
            total_samples += images.size(0)

    avg_val_loss = running_val_loss / len(dataloader)

    # Compute mean absolute error and accuracy
    mae_sphere = total_sphere_error / total_samples
    mae_cylinder = total_cylinder_error / total_samples

    # Accuracy as 100 - MAE for simplicity in presentation
    accuracy_sphere = 100 - mae_sphere
    accuracy_cylinder = 100 - mae_cylinder
    avg_accuracy = (accuracy_sphere + accuracy_cylinder) / 2

    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Mean Absolute Error (Sphere): {mae_sphere:.4f}")
    logging.info(f"Mean Absolute Error (Cylinder): {mae_cylinder:.4f}")
    logging.info(f"Sphere Accuracy: {accuracy_sphere:.2f}%")
    logging.info(f"Cylinder Accuracy: {accuracy_cylinder:.2f}%")
    logging.info(f"Overall Accuracy: {avg_accuracy:.2f}%")

    return avg_val_loss, avg_accuracy

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dataset_path = './Choithram Netralaya Data/Images'
    images_metadata = load_images_from_directory(dataset_path)
    mapped_images = map_images_to_acuity(images_metadata, acuity_data)

    valid_mapped_images = validate_data(mapped_images)
    dataloader = get_dataloader(valid_mapped_images, batch_size=16, num_workers=6)

    model = initialize_model()
    criterion, optimizer = get_loss_and_optimizer(model)

    train_model(dataloader, model, criterion, optimizer, epochs=30, early_stopping_patience=7)

    # Evaluate the model after training
    logging.info("Evaluating model...")
    _, final_accuracy = validate_model(dataloader, model, criterion)
    logging.info(f"Final accuracy of the model: {final_accuracy:.2f}%")


