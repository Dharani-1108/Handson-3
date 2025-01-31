import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the fully connected (fc) layer to output 7 classes
num_emotions = 7
model.fc = nn.Linear(model.fc.in_features, num_emotions)

# Set the model to evaluation mode
model.eval()

# Define emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define image folder
image_folder = "/content/fer2013/fer2013/images/train/happy"  # Adjust this path as needed

# Confirm the image folder exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder not found at {image_folder}")

# Get a random image from the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
if not image_files:
    raise ValueError("No image files found in the specified folder!")

random_image_path = random.choice(image_files)
print(f"Selected random image: {random_image_path}")

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels for ResNet
    transforms.Resize((224, 224)),               # Resize to ResNet's input size
    transforms.ToTensor(),                       # Convert to tensor
])

# Load and preprocess the random image
image = Image.open(random_image_path)
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)                # Forward pass
    predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class index

# Map the predicted index to emotion
predicted_emotion = emotions[predicted_class]

# Display the result
print(f"Predicted Emotion: {predicted_emotion}")

