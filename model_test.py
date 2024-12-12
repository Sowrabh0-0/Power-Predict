import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


# Load the saved model
def load_model(model_path, device):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2)

    # Load the model's weights only
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


# Preprocessing transformations for grayscale (IR) images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as for RGB images
])


# Function to predict sphere and cylinder values
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('L')  # Load as grayscale
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(input_image)
        predicted_sphere = outputs[0, 0].item()
        predicted_cylinder = outputs[0, 1].item()

    return predicted_sphere, predicted_cylinder


# Function to display the image and predictions
def display_image_with_prediction(image_path, predicted_sphere, predicted_cylinder, eye_type):
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')  # Display the image as grayscale
    plt.axis('off')
    plt.title(f"Predicted Sphere ({eye_type}): {predicted_sphere:.2f}, Cylinder ({eye_type}): {predicted_cylinder:.2f}")
    plt.show()


# Function to load IR images only from folders "1", "2", and "3"
def load_images_from_directory(directory_path, allowed_folders=['1', '2', '3'],
                               image_extensions=['.jpg', '.jpeg', '.png']):
    images_metadata = []
    for folder_name in allowed_folders:
        folder_path = os.path.join(directory_path, folder_name)
        if not os.path.exists(folder_path):
            continue  # Skip if the folder doesn't exist
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if "_IR" in file and ("LEFT" in file or "RIGHT" in file):  # Check if it's an IR image
                        file_path = os.path.join(root, file)
                        eye_type = "OD" if "RIGHT_IR" in file else "OS"  # Determine Right Eye (OD) or Left Eye (OS)
                        images_metadata.append({
                            'filename': file,
                            'eye_type': eye_type,
                            'path': file_path
                        })
    return images_metadata


if __name__ == "__main__":
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = "./custom_model.pth"  # Path to the saved model
    model = load_model(model_path, device)

    # Directory containing images
    dataset_path = "./ODOCS RED REFLEX DATABASE/Choithram Netralaya Data/Images"

    # Load the images from folders "1", "2", and "3" within the directory
    images_metadata = load_images_from_directory(dataset_path)

    # Iterate through all images and predict for each one
    for image_meta in images_metadata:
        image_path = image_meta['path']
        eye_type = image_meta['eye_type']

        # Print which file is being processed
        print(f"Opening file: {image_meta['filename']} (Eye Type: {eye_type})")

        # Predict sphere and cylinder values
        predicted_sphere, predicted_cylinder = predict_image(image_path, model, device)
        print(
            f"Predicted Sphere ({eye_type}): {predicted_sphere:.2f}, Predicted Cylinder ({eye_type}): {predicted_cylinder:.2f}")

        # Display image with predictions
        display_image_with_prediction(image_path, predicted_sphere, predicted_cylinder, eye_type)
