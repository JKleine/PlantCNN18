import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# Load saved model
saved_model = torch.load('model_epoch9.pt')
model_state_dict = saved_model['model_state_dict']

# Create model architecture
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)

# Load model state dict
model.load_state_dict(model_state_dict)

# Set model to evaluation mode
model.eval()

# Define transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get list of all images in the test folder
test_folder = 'test'
image_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.jpg')]

# Iterate over all images and make predictions
for image_path in image_paths:
    # Load input image and apply transform
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)

    # Get predicted class
    _, predicted = torch.max(output.data, 1)
    class_index = predicted.item()

    # Load label names
    with open('labels.txt') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    # Print predicted class label
    print(f'{image_path}: {labels[class_index]}')
