import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(self, excel_file_path, dataset_path):
        self.excel_file_path = excel_file_path
        self.dataset_path = dataset_path
        self.acuity_data = self._load_acuity_data()
        self.image_extensions = ['.jpg', '.jpeg', '.png']
        self.transform = self._get_transforms()

    def _load_acuity_data(self):
        try:
            data = pd.read_excel(self.excel_file_path)
            logging.info("Acuity data loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading acuity data: {e}")
            raise

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
            transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_images(self):
        images_metadata = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
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
        logging.info(f"Loaded metadata for {len(images_metadata)} images.")
        return images_metadata

    def map_to_acuity(self, images_metadata):
        mapped_data = []
        for meta in images_metadata:
            folder_name = os.path.basename(os.path.dirname(meta['path']))
            try:
                patient_id = int(folder_name)
            except ValueError:
                logging.warning(f"Invalid patient ID: {folder_name}. Skipping.")
                continue
            eye = meta['eye_type']
            patient_data = self.acuity_data[self.acuity_data['patient'] == patient_id]
            if not patient_data.empty:
                if eye == 'OD':
                    meta['sphere'] = patient_data['r sphere'].values[0]
                    meta['cylinder'] = patient_data['r cylinder'].values[0]
                else:
                    meta['sphere'] = patient_data['l sphere'].values[0]
                    meta['cylinder'] = patient_data['l cylinder'].values[0]
                mapped_data.append(meta)
        logging.info(f"Mapped {len(mapped_data)} images to acuity data.")
        return mapped_data

    def validate_data(self, mapped_data):
        valid_images = []
        valid_spheres = [float(meta['sphere']) for meta in mapped_data if pd.notna(meta['sphere'])]
        valid_cylinders = [float(meta['cylinder']) for meta in mapped_data if pd.notna(meta['cylinder'])]

        mean_sphere = sum(valid_spheres) / len(valid_spheres) if valid_spheres else 0.0
        mean_cylinder = sum(valid_cylinders) / len(valid_cylinders) if valid_cylinders else 0.0

        for meta in mapped_data:
            try:
                sphere = float(meta['sphere']) if pd.notna(meta['sphere']) else mean_sphere
                cylinder = float(meta['cylinder']) if pd.notna(meta['cylinder']) else mean_cylinder
                meta['sphere'] = sphere
                meta['cylinder'] = cylinder
                valid_images.append(meta)
            except ValueError:
                continue

        logging.info(f"Validated {len(valid_images)} images.")
        return valid_images

    def process_data(self):
        images_metadata = self.load_images()
        mapped_data = self.map_to_acuity(images_metadata)
        return self.validate_data(mapped_data)


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
