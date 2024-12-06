import json
import os
import torch
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelScorer:
    def __init__(self, model, dataloader, criterion, score_file='model_scores.json', models_dir='models'):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.score_file = score_file
        self.models_dir = models_dir

        os.makedirs(self.models_dir, exist_ok=True)

    def validate(self):
        """Run validation and calculate metrics."""
        self.model.eval()
        running_val_loss = 0.0
        total_sphere_error = 0.0
        total_cylinder_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, spheres, cylinders in self.dataloader:
                images = images.to('cuda')
                spheres = spheres.to('cuda')
                cylinders = cylinders.to('cuda')

                outputs = self.model(images)
                predicted_spheres = outputs[:, 0]
                predicted_cylinders = outputs[:, 1]

                loss_sphere = self.criterion(predicted_spheres, spheres)
                loss_cylinder = self.criterion(predicted_cylinders, cylinders)
                loss = loss_sphere + loss_cylinder
                running_val_loss += loss.item()

                total_sphere_error += torch.sum(torch.abs(predicted_spheres - spheres)).item()
                total_cylinder_error += torch.sum(torch.abs(predicted_cylinders - cylinders)).item()
                total_samples += len(images)

        avg_val_loss = running_val_loss / len(self.dataloader)
        mae_sphere = total_sphere_error / total_samples
        mae_cylinder = total_cylinder_error / total_samples

        sphere_accuracy = 100 - mae_sphere
        cylinder_accuracy = 100 - mae_cylinder
        overall_accuracy = (sphere_accuracy + cylinder_accuracy) / 2

        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Mean Absolute Error (Sphere): {mae_sphere:.4f}")
        logging.info(f"Mean Absolute Error (Cylinder): {mae_cylinder:.4f}")
        logging.info(f"Sphere Accuracy: {sphere_accuracy:.2f}%")
        logging.info(f"Cylinder Accuracy: {cylinder_accuracy:.2f}%")
        logging.info(f"Overall Accuracy: {overall_accuracy:.2f}%")

        return {
            "validation_loss": avg_val_loss,
            "mae_sphere": mae_sphere,
            "mae_cylinder": mae_cylinder,
            "sphere_accuracy": sphere_accuracy,
            "cylinder_accuracy": cylinder_accuracy,
            "overall_accuracy": overall_accuracy,
        }

    def save_epoch_score(self, epoch, metrics):
        """Save metrics for each epoch to the JSON file."""
        metrics["epoch"] = epoch
        metrics["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["best"] = False  

        try:
            if not os.path.exists(self.score_file):
                with open(self.score_file, 'w') as f:
                    json.dump([metrics], f, indent=4)
            else:
                with open(self.score_file, 'r+') as f:
                    data = json.load(f)
                    data.append(metrics)
                    f.seek(0)
                    json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save epoch metrics: {e}")

    def finalize_best_score(self):
        """Determine the best score and save the best model."""
        try:
            if not os.path.exists(self.score_file):
                logging.warning("Score file does not exist. No best score to finalize.")
                return

            with open(self.score_file, 'r+') as f:
                data = json.load(f)

                best_entry = min(data, key=lambda x: x["validation_loss"])
                for entry in data:
                    entry["best"] = entry == best_entry

                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            best_model_path = os.path.join(self.models_dir, f"best_model_{timestamp}.pth")
            torch.save(self.model.state_dict(), best_model_path)
            logging.info(f"Best model saved to {best_model_path}")

        except Exception as e:
            logging.error(f"Failed to finalize best score: {e}")
