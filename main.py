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