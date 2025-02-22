# PyTorch DataLoader Examples and Exercises

Let’s expand on PyTorch `DataLoader` with two new examples: numerical data using Pandas and image data using PIL or torchvision.

---

## Example 1: Numerical Data with Pandas

When working with numerical data (e.g., tabular data like CSV files), you can use Pandas to load and preprocess it, then wrap it in a `Dataset` for PyTorch.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Custom Dataset for Numerical Data
class NumericalDataset(Dataset):
    def __init__(self, csv_file):
        # Load CSV with Pandas
        self.data = pd.read_csv(csv_file)
        # Assume last column is the target, others are features
        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Simulate a CSV file (in real use, replace with actual file path)
data = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'label'])
data.to_csv('dummy_numerical.csv', index=False)

# Create Dataset and DataLoader
dataset = NumericalDataset('dummy_numerical.csv')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Move to CUDA and iterate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch_idx, (features, labels) in enumerate(dataloader):
    features, labels = features.to(device), labels.to(device)
    print(f"Batch {batch_idx + 1}: Features shape: {features.shape}, Labels shape: {labels.shape}")
    break  # Just one batch for demo
```

### Explanation
- **Pandas**: Loads the CSV into a DataFrame. Features are all columns except the last; labels are the last column.
- **Tensor Conversion**: Converts DataFrame values to PyTorch tensors with appropriate dtype (`float32`).
- **CUDA**: `.to(device)` moves the batched tensors (`features` and `labels`) to GPU if available.

---

## Example 2: Image Data with torchvision

For image data, you can use `torchvision.datasets` (e.g., for built-in datasets like MNIST) or create a custom dataset with PIL or OpenCV to load images from a folder.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Custom Dataset for Images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
        # Dummy labels (e.g., filename-based or from a CSV)
        self.labels = [int(img.split('_')[1].split('.')[0]) for img in os.listdir(image_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transform (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Assume a folder 'images' with files like 'img_0.png', 'img_1.png', etc.
# For demo, create dummy images if folder doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
    for i in range(5):
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(f'images/img_{i}.png')

# Create Dataset and DataLoader
dataset = ImageDataset('images', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Move to CUDA and iterate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch_idx, (images, labels) in enumerate(dataloader):
    images, labels = images.to(device), labels.to(device)
    print(f"Batch {batch_idx + 1}: Images shape: {images.shape}, Labels shape: {labels.shape}")
    break  # Just one batch for demo
```

### Explanation
- **Images**: Loaded with PIL, transformed (resized, normalized) using `torchvision.transforms`.
- **Labels**: Extracted from filenames (dummy example; in practice, use a CSV or folder structure).
- **CUDA**: Batched `images` (e.g., shape `[2, 3, 32, 32]`) and `labels` moved to GPU.

---

## Exercises

Here are some exercises to practice `DataLoader` concepts, with answers below.

### Exercise 1: Numerical Data
Create a `Dataset` for a CSV with 3 feature columns and 1 label column. Use a batch size of 4 and print the first batch’s shape.

### Exercise 2: Image Data
Modify the image example to handle grayscale images (1 channel instead of 3). Adjust the transform and print the batch shape.

### Answers

#### Answer 1: Numerical Data
```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class NumericalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.features[index], self.labels[index]

# Dummy CSV
data = pd.DataFrame(np.random.rand(10, 4), columns=['f1', 'f2', 'f3', 'label'])
data.to_csv('exercise1.csv', index=False)

dataset = NumericalDataset('exercise1.csv')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for features, labels in dataloader:
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")  # [4, 3], [4]
    break
```

#### Answer 2: Image Data (Grayscale)
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class GrayscaleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
        self.labels = [int(img.split('_')[1].split('.')[0]) for img in os.listdir(image_dir) if img.endswith('.png')]

    def __len__(self): return len(self.images)
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L')  # 'L' for grayscale
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.exists('grayscale_images'):
    os.makedirs('grayscale_images')
    for i in range(5):
        Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(f'grayscale_images/img_{i}.png')

dataset = GrayscaleImageDataset('grayscale_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for images, labels in dataloader:
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")  # [2, 1, 32, 32], [2]
    break
```

---

## Python Script to Test Your Knowledge

Here’s a script that quizzes you on `DataLoader` concepts. Run it, answer the questions, and it’ll check your responses.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class QuizDataset(Dataset):
    def __init__(self):
        self.questions = [
            "What does __len__ return? (a) A tensor, (b) An integer, (c) A tuple",
            "What does __getitem__ typically return? (a) Batch size, (b) Single sample tuple, (c) Dataset size",
            "How do you move a tensor to GPU? (a) .cuda(), (b) .to(device), (c) Both a and b"
        ]
        self.answers = ['b', 'b', 'c']

    def __len__(self): return len(self.questions)
    def __getitem__(self, index): return self.questions[index], self.answers[index]

def run_quiz():
    dataset = QuizDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    score = 0

    print("Welcome to the PyTorch DataLoader Quiz!")
    for i, (question, answer) in enumerate(dataloader):
        print(f"\nQuestion {i + 1}: {question[0]}")
        user_answer = input("Your answer (a/b/c): ").strip().lower()
        correct_answer = answer[0]
        if user_answer == correct_answer:
            print("Correct!")
            score += 1
        else:
            print(f"Wrong! Correct answer: {correct_answer}")
    print(f"\nQuiz complete! Score: {score}/{len(dataset)}")

if __name__ == "__main__":
    run_quiz()
```

### How to Use
1. Save this as `dataloader_quiz.py`.
2. Run it: `python dataloader_quiz.py`.
3. Answer the questions (type `a`, `b`, or `c`). It’ll tell you if you’re right and show your score.

---

## Final Notes
- **Numerical Data**: Use Pandas for CSVs, convert to tensors.
- **Image Data**: Use PIL/`torchvision` for loading and transforming images.
- Practice with the exercises and quiz to solidify your understanding!

Let me know if you’d like more examples or tweaks!
```

---

### What You Get
- **Examples**: Numerical data with Pandas and image data with `torchvision`.
- **Exercises**: Two practical tasks with solutions.
- **Quiz Script**: A runnable Python file to test your knowledge interactively.
