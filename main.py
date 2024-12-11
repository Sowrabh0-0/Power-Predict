import torch
from torch.utils.data import DataLoader
from data_pipeline import DataPipeline, RedReflexDataset
# from model_train import ModelTrainer
from custom_model import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA is not available. Using CPU.")

    excel_file_path = '/home/rage/Downloads/dataset/ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/acuityvalues.xlsx'
    dataset_path = '/home/rage/Downloads/dataset/ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/Images'

    try:
        pipeline = DataPipeline(excel_file_path, dataset_path)
        valid_mapped_images = pipeline.process_data()

        dataset = RedReflexDataset(valid_mapped_images, transform=pipeline.transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

        trainer = ModelTrainer(dataloader)
        trainer.train(epochs=30)

    except KeyboardInterrupt:
        logging.info("Training interrupted! Saving current model state...")
        trainer.scorer.save_best_model(trainer.best_metrics)
        logging.info("Model and metrics saved successfully. Exiting gracefully.")
