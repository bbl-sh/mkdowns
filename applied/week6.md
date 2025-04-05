# Week 6 (Optimizing Deep Learning Training: Automatic Mixed Precision, Transfer Learning)

**Optimizing Deep Learning Training: Automatic Mixed Precision**

I’m learning about Automatic Mixed Precision (AMP) from this PDF. It starts by talking about optimizing deep learning training, which means making neural networks better by tuning them to lower losses and get good accuracy. This tuning can involve things like learning rate and weights. An example is the Gradient Descent algorithm, which helps adjust the network step by step. There are different types like Stochastic Gradient Descent, Mini-Batch Gradient Descent, and Momentum, which helps fix some issues in Stochastic Gradient Descent.

Then it moves to advanced stuff like using less memory, faster data transfer, and quick calculations with something called Tensor Cores. Some ways to do this are using GPUs, Mixed Precision, XLA, and Transfer Learning. Mixed Precision is about using two number formats together: FP32 (32-bit, single precision) and FP16 (16-bit, half precision) during training. This works on NVIDIA GPUs like Ampere, Volta, and Turing because they have Tensor Cores.

Tensor Cores make math operations like matrix multiplication and convolutions faster with FP16. The benefits of Mixed Precision are that it speeds up these operations, uses less memory bandwidth so data moves quicker, and needs less memory overall, letting us train bigger networks. On Volta and Turing GPUs, it can make things 3 times faster.

Mixed Precision Training means doing most calculations in FP16 to speed things up, but keeping some important parts in FP32 to not lose info. The steps are: change the model to use FP16 where it’s okay, and add loss scaling to keep small gradient values from disappearing. When using FP16, some gradients can become too tiny or zero because FP16 can’t handle very small numbers well. Scaling shifts these gradients into a range FP16 can work with, matching FP32 accuracy.

Loss scaling works by multiplying the loss by a big number before backpropagation, then dividing the gradients by that number before updating weights. This keeps the updates the same as in FP32. The PDF shows a process: keep weights in FP32, make an FP16 copy, do forward and backward passes in FP16, scale the loss, unscale gradients, and update weights.

Choosing the scaling factor can be tricky. You can pick a constant one based on gradient stats or try different values. It should keep the biggest gradient times the factor below 65,504 to avoid overflow in FP16. There’s also a dynamic way: start with a big factor, lower it if gradients show infinities or NaNs, and raise it if things go well for a while.

In summary, Mixed Precision Training picks FP16 for Tensor Core ops, scales the loss, does backpropagation with scaled gradients, unscales them, and updates weights. But this can be hard to manage, so Automatic Mixed Precision (AMP) makes it easier. AMP auto-converts some operations to 16-bit for speed and handles everything for you. It works in frameworks like MXNet, PyTorch, and TensorFlow with scripts or containers from NVIDIA NGC. The PDF shows code for TensorFlow (using an environment variable or graph rewrite) and PyTorch (using GradScaler and autocast).

**Transfer Learningf**

Transfer Learning. It’s a way to use a model already trained on lots of data to help with a new task. Instead of starting fresh, you take a pretrained model, use its weights and settings, and train it on your new data to make a new model. You can tweak the settings for better accuracy.

Why use it? You don’t need to build a model from nothing, it needs less data, trains faster, and saves money. It’s useful in areas like Image Classification (figuring out what’s in a picture), Object Detection (finding objects), Segmentation (splitting images into parts), ASR (speech to text), NLP (language tasks), and Computational AI.

NVIDIA’s TAO (Train, Adapt, Optimize) helps with Transfer Learning. It has pretrained models in two types: purpose-built (for specific jobs) and general-purpose (for many tasks). The TAO Toolkit has containers for computer vision steps like data prep and training. For image classification, it talks about VGG-16, a 16-layer model from 2014 with 138 million parameters, trained on 4 NVIDIA GPUs, often used to extract features. Then there’s ResNet-50, a deeper 50-layer model from 2015 with 23 million parameters and skip connections, making it more accurate.

TAO focuses on apps like the TAO Toolkit, YOLOV4, and YOLOV5. For image classification with ResNet-18, it uses datasets like Kitti or TFrecord. There are two spec files: classification_spec.cfg and classification_retrain_spec.cfg, with sections for model setup, training, and evaluation. YOLOV5 is a set of object detection models pretrained on COCO. To train it on custom data, you prepare a dataset with test, train, and valid folders, label it with tools like Roboflow, and make a data.yaml file with paths and class details.

In short, Transfer Learning uses ready models to save time and effort, and TAO makes it simple for vision tasks with models like VGG-16, ResNet, and YOLOV5.


# Codes used in this week

### 1. Automatic Mixed Precision (AMP) in PyTorch

```python
import torch
from torch.cuda.amp import GradScaler, autocast

model = torch.nn.Linear(10, 2).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = GradScaler()

for epoch in range(5):
    data = torch.randn(5, 10).cuda()
    target = torch.randint(0, 2, (5,)).cuda()
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

```

---

### 2. Transfer Learning with Pre-trained ResNet-18 in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Freeze all the layers except last
for param in model.parameters():
    param.requires_grad = False
# layer just before it in the ResNet-18 architecture, which is layer4.
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder('data/train', transform=transform)
val_dataset = datasets.ImageFolder('data/val', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')

```
