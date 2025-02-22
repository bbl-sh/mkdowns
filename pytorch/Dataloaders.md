
# PyTorch DataLoader Basics

Let’s dive into PyTorch `DataLoader` basics! I’ll explain what it does, the key functions involved, what they return, and then provide a full example .

---

## Basics of PyTorch DataLoader

The `DataLoader` in PyTorch is a utility that helps you efficiently load and batch your data for training machine learning models.

To use a `DataLoader`, you typically need three main components from a `Dataset`:

1. `__init__`: Initializes the dataset (e.g., sets up file paths or data).
2. `__len__`: Returns the total size of the dataset.
3. `__getitem__`: Fetches a single data sample (e.g., an image and its label).

Let’s break these down and see what they return, then move to a full code example with CUDA.

---

## The Three Key Functions and Their Returns

### 1. `__init__(self, ...)`
- **Purpose**: Sets up the dataset by storing things like file paths, data arrays, or labels. It’s called once when you create the dataset object.
- **Returns**: Nothing (it’s a constructor, so it just initializes the object).
- **Example**: If you’re loading images, `__init__` might take a folder path and build a list of image filenames and labels.

### 2. `__len__(self)`
- **Purpose**: Tells the `DataLoader` how many samples are in the dataset. This is crucial for things like progress bars or determining epochs.
- **Returns**: An integer representing the total number of samples.
- **Example**: If you have 1,000 images, `__len__` returns `1000`.

### 3. `__getitem__(self, index)`
- **Purpose**: Retrieves a single sample (e.g., one image and its label) based on the given `index`. The `DataLoader` calls this repeatedly to fetch data.
- **Returns**: Typically a tuple containing the data and its target (e.g., `(image_tensor, label)`). The exact return type depends on your data.
- **Example**: For an image dataset, it might return a tensor of pixel values and an integer label.

The `DataLoader` wraps this `Dataset` and iterates over it, grouping samples into batches (controlled by the `batch_size` parameter) and optionally shuffling them.

---

## Full PyTorch Code Example with CUDA

Now, let’s put it all together with a simple example: a dataset of dummy images (random tensors) and labels, loaded with a `DataLoader`, and moved to CUDA (GPU). I’ll explain each part and highlight how data gets to the GPU.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Define a Custom Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Initialize with some dummy data: 1000 "images" of size 3x32x32 (channels, height, width)
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 32, 32)  # Random "images"
        self.labels = torch.randint(0, 10, (num_samples,))  # Random labels (0-9)

    def __len__(self):
        # Return the total number of samples
        return self.num_samples

    def __getitem__(self, index):
        # Return a single sample: image and label
        image = self.images[index]
        label = self.labels[index]
        return image, label

# Step 2: Create Dataset and DataLoader
dataset = DummyDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Step 3: Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 4: Training Loop Example with CUDA
for epoch in range(2):  # 2 epochs for demo
    print(f"Epoch {epoch + 1}")
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to CUDA (GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Simulate some "processing" (e.g., a model forward pass)
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
        print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}, device: {labels.device}")

        # Here, you'd normally pass `images` to a model, compute loss, etc.
        # For simplicity, we just print and move on

# Optional: Clear GPU memory (good practice in notebooks)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## Explanation of the Code

### Dataset Setup
- **`DummyDataset`**: Creates 1,000 random “images” (tensors of shape `[3, 32, 32]`) and labels (integers 0-9).
- **`__len__`**: Returns `1000`.
- **`__getitem__`**: Returns a tuple `(image, label)` for the given index, where `image` is a tensor of shape `[3, 32, 32]` and `label` is a scalar tensor.

### DataLoader
- **`DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)`**:
  - `batch_size=32`: Groups 32 samples into one batch.
  - `shuffle=True`: Randomizes the order of samples each epoch.
  - `num_workers=2`: Uses 2 subprocesses to load data in parallel (speeds things up).

When you iterate over the `DataLoader`, it yields batches. Each batch is a tuple `(batch_images, batch_labels)`:
- `batch_images`: Shape `[32, 3, 32, 32]` (32 images, each 3x32x32).
- `batch_labels`: Shape `[32]` (32 labels).

### Moving to CUDA
- **`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`**:
  - Checks if a GPU is available. If not, it falls back to CPU.
- **`images.to(device)` and `labels.to(device)`**:
  - The `.to(device)` method moves tensors from CPU memory (default) to GPU memory (if `device` is `"cuda"`).
  - **What’s passed to CUDA**: The `images` tensor (shape `[32, 3, 32, 32]`) and `labels` tensor (shape `[32]`).
  - **How it’s passed**: PyTorch handles this internally by copying the tensor data to GPU memory. The tensor’s `.device` attribute changes from `cpu` to `cuda:0` (or another GPU index if you have multiple GPUs).

### Output
Running this on a GPU-enabled system might print something like:
```
Using device: cuda
Epoch 1
Batch 1:
  Images shape: torch.Size([32, 3, 32, 32]), dtype: torch.float32, device: cuda:0
  Labels shape: torch.Size([32]), dtype: torch.int64, device: cuda:0
...
```

---

## Key Takeaways
- **`__init__`**: Sets up data, returns nothing.
- **`__len__`**: Returns an integer (dataset size).
- **`__getitem__`**: Returns a sample (e.g., `(image, label)`).
- **`DataLoader`**: Batches and shuffles data, yielding `(batch_images, batch_labels)`.
- **CUDA**: Use `.to(device)` to move tensors to the GPU. PyTorch manages the memory transfer.
