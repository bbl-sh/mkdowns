# Understanding PyTorch Dataset: A Beginner's Guide

In PyTorch, a `Dataset` is an abstract class that represents a dataset. It provides a way to access individual data points (samples) and their corresponding labels. The `DataLoader`, which we covered earlier, works on top of the `Dataset` to handle batching, shuffling, and parallel loading.

In this guide, we'll cover:
1. What is a Dataset?
2. How to create a custom Dataset.
3. Examples of using built-in Datasets for numerical and image data.
4. Two practical examples with solutions.
5. Exercises to practice what you've learned.

---

## 1. What is a Dataset?

A **Dataset** in PyTorch is any object that implements two key methods:
- `__len__`: Returns the size of the dataset.
- `__getitem__`: Retrieves a single data point (and its label) by index.

PyTorch provides several built-in datasets (e.g., MNIST, CIFAR-10), but you can also create your own custom datasets by subclassing `torch.utils.data.Dataset`.

### Basic Syntax:
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # Initialize your dataset here
        pass

    def __len__(self):
        # Return the total number of samples
        pass

    def __getitem__(self, idx):
        # Return the sample at index `idx`
        pass
```

---

## 2. Creating a Custom Dataset

Let's create a simple custom dataset where each sample is a random number, and the label is the square of that number.

### Example 1: Custom Dataset for Numerical Data

```python
import torch
from torch.utils.data import Dataset

class SquareDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randint(0, 100, (num_samples,))  # Random numbers between 0 and 100
        self.labels = self.data ** 2  # Labels are squares of the numbers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Instantiate the dataset
dataset = SquareDataset(num_samples=10)

# Access a single sample
sample, label = dataset[0]
print("Sample:", sample.item(), "Label:", label.item())
```

### Output:
```
Sample: 42 Label: 1764
```

---

## 3. Using Built-in Datasets for Numerical and Image Data

PyTorch provides several built-in datasets via the `torchvision.datasets` module, such as MNIST, CIFAR-10, and more. These datasets are commonly used for image classification tasks.

### Example 2: Built-in Dataset for Image Data (MNIST)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Access a single sample
image, label = mnist_train[0]
print("Image Shape:", image.shape)
print("Label:", label)
```

### Output:
```
Image Shape: torch.Size([1, 28, 28])
Label: 5
```

---

## 4. Two Practical Examples with Solutions

### Example 1: Custom Dataset for Regression Task

Suppose you want to predict house prices based on some numerical features like the number of rooms and area.

#### Solution:

```python
import torch
from torch.utils.data import Dataset

class HousePriceDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([[3, 1200], [2, 800], [4, 1500], [3, 1000]], dtype=torch.float32)
        self.labels = torch.tensor([300000, 200000, 400000, 250000], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Instantiate the dataset
dataset = HousePriceDataset()

# Access a single sample
features, label = dataset[0]
print("Features:", features, "Label:", label)
```

### Output:
```
Features: tensor([   3., 1200.]) Label: tensor(300000.)
```

### Example 2: Custom Dataset for Image Classification Task

Let's create a custom dataset for grayscale images of handwritten digits.

#### Solution:

```python
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage
image_paths = ['./data/image1.png', './data/image2.png']  # Replace with actual paths
labels = [0, 1]  # Corresponding labels

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomImageDataset(image_paths, labels, transform=transform)

# Access a single sample
image, label = dataset[0]
print("Image Shape:", image.shape)
print("Label:", label)
```

### Output:
```
Image Shape: torch.Size([1, 28, 28])
Label: 0
```

---

## 5. Exercises to Practice

Now that you understand how to create and use datasets in PyTorch, here are some exercises to help you practice:

### Exercise 1: Custom Dataset for Numerical Data
- Create a custom dataset where each sample is a random number between 0 and 100, and the label is the cube of that number.
- Use the dataset to retrieve the first 5 samples and print them.

#### Solution:

```python
import torch
from torch.utils.data import Dataset

class CubeDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randint(0, 100, (num_samples,))
        self.labels = self.data ** 3

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Instantiate the dataset
dataset = CubeDataset(num_samples=5)

# Print the first 5 samples
for i in range(len(dataset)):
    sample, label = dataset[i]
    print(f"Sample {i}: {sample.item()}, Label: {label.item()}")
```

### Exercise 2: Custom Dataset for Image Data
- Create a custom dataset for RGB images stored in a folder. Each image should be resized to 64x64 and normalized.
- Use the dataset to retrieve the first image and print its shape.

#### Solution:

```python
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RGBImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Example usage
image_paths = ['./data/rgb_image1.jpg', './data/rgb_image2.jpg']  # Replace with actual paths

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = RGBImageDataset(image_paths, transform=transform)

# Access the first image
image = dataset[0]
print("Image Shape:", image.shape)
```

### Exercise 3: Real-world Dataset
- Download the CIFAR-10 dataset using `torchvision.datasets`.
- Apply transformations to resize the images to 32x32 and normalize them.
- Use a `DataLoader` to load the data in batches of 16 and print the shape of the first batch.

#### Solution:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader
train_loader = DataLoader(cifar_train, batch_size=16, shuffle=True)

# Iterate through the DataLoader
for images, labels in train_loader:
    print("Batch Images Shape:", images.shape)
    print("Batch Labels:", labels)
    break
```

---
