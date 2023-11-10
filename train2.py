import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("train", transform=train_transforms)
val_dataset = datasets.ImageFolder("val", transform=val_transforms)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 20)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("Training has begun.")


# Train function
def train(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_predictions += torch.sum(predictions == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions.double() / len(train_loader.dataset)

    return epoch_loss, epoch_accuracy

# Validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(predictions == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = correct_predictions.double() / len(val_loader.dataset)

    return epoch_loss, epoch_accuracy

# Main function
if __name__ == '__main__':
    for epoch in range(10):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)

        # Print training and validation metrics
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")

       # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, 'model_epoch{}.pt'.format(epoch))

