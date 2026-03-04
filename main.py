import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Transform: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load Pokemon dataset from PokemonData folder
full_dataset = torchvision.datasets.ImageFolder(root='./PokemonData', transform=transform)


# Split dataset: 80% train, 20% test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Set seed for reproducibility
torch.manual_seed(42)
trainset, testset = random_split(full_dataset, [train_size, test_size])

# Create data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Get classes (Pokemon names)
classes = full_dataset.classes
print(f"Number of Pokemon types: {len(classes)}")
print(f"Training samples: {train_size}, Test samples: {test_size}")
print("First 10 Pokemon types:", classes[:10])


# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images[:8]))
print('Labels:', [classes[labels[j].item()] for j in range(8)])

# CNN
# Pass the number of classes into the model dynamically
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # Updated from 2048 to 8192 based on the 64x64 input size
            nn.Linear(8192, 128),
            nn.ReLU(),
            # Updated to use dynamic number of classes instead of a hardcoded 10
            nn.Linear(128, num_classes) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Instantiate the model with the correct number of Pokemon classes
num_pokemon_classes = len(classes)
cnn = SimpleCNN(num_classes=num_pokemon_classes)
print(cnn)

# Instructions: Train the CNN similar to the previous network. Test accuracy should improve.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/200:.3f}")
            running_loss = 0.0

print("Finished Training CNN")

# Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the CNN on test images: {100 * correct / total:.2f}%')
print('Labels:', [classes[labels[j].item()] for j in range(8)])

# CNN
# Pass the number of classes into the model dynamically
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # Updated from 2048 to 8192 based on the 64x64 input size
            nn.Linear(8192, 128),
            nn.ReLU(),
            # Updated to use dynamic number of classes instead of a hardcoded 10
            nn.Linear(128, num_classes) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Instantiate the model with the correct number of Pokemon classes
num_pokemon_classes = len(classes)
cnn = SimpleCNN(num_classes=num_pokemon_classes)
print(cnn)

# Instructions: Train the CNN similar to the previous network. Test accuracy should improve.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/200:.3f}")
            running_loss = 0.0

print("Finished Training CNN")

# Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the CNN on test images: {100 * correct / total:.2f}%')