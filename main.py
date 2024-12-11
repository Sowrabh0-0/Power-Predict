import torch
from torch.utils.data import DataLoader
from data_pipeline import DataPipeline, RedReflexDataset
from custom_model import ModelTrainer
import logging
import traceback
import sys
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_memory_info():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 2)
        cached_memory = torch.cuda.memory_reserved(0) / (1024 ** 2)
        logging.info(f"Total GPU Memory: {total_memory:.2f} MB")
        logging.info(f"Allocated GPU Memory: {allocated_memory:.2f} MB")
        logging.info(f"Cached GPU Memory: {cached_memory:.2f} MB")

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print_memory_info()
    else:
        logging.info("CUDA is not available. Using CPU.")

    excel_file_path = '/home/rage/Downloads/dataset/ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/acuityvalues.xlsx'
    dataset_path = '/home/rage/Downloads/dataset/ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/Images'

    try:
        pipeline = DataPipeline(excel_file_path, dataset_path)
        valid_mapped_images = pipeline.process_data()

        if not valid_mapped_images:
            logging.error("No valid images found in the dataset!")
            sys.exit(1)

        dataset = RedReflexDataset(valid_mapped_images, transform=pipeline.transform)
        logging.info(f"Dataset created with {len(dataset)} samples")

        dataloader = DataLoader(
            dataset, 
            batch_size=8,  # Reduced from 16 to 8
            shuffle=True, 
            num_workers=2,  # Reduced from 6 to 2
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        logging.info(f"Dataloader created with {len(dataloader)} batches")

        trainer = ModelTrainer(dataloader)
        trainer.train(epochs=30)

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        traceback.print_exc()
        sys.exit(1)