# Week 4 (Pytorch, Distributed Training)

### Introduction to PyTorch

I will start with the basics of pytorch as covered in the course and then there is codes for Distributed Training, i have added some codes too for my personal preference, let's go(Chama)

#### PyTorch Tensor

Tensors are like NumPy arrays but work on GPUs too.

```python
import torch
x = torch.rand(2, 3)
y = torch.rand(5, 3)
print(x)
print(y)
```
This code creates two random tensors: `x` with shape (2,3) and `y` with shape (5,3), then prints them.

```python
x = torch.zeros(5, 3)
x = torch.tensor([5.5, 3])
```
This makes a tensor `x` of zeros with shape (5,3), then reassigns `x` to a tensor with values [5.5, 3].

```python
print(x.size())
```
This prints the shape of tensor `x`.

#### Operations

```python
y = torch.rand(5, 3)
print(x + y)
```
This adds tensors `x` and `y` element-wise (if shapes match).

```python
print(x[:, 1])
```
This gets the second column of tensor `x`.

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
```
This reshapes tensor `x` (4,4): `y` becomes 1D with 16 elements, `z` becomes 2D with 2 rows and 8 columns. It prints their shapes.

```python
x = torch.randn(1, 4, 32, 24)
y = x.view(8, 2, -1, 3, 8)
print(y.size())
```
This reshapes a tensor `x` into `y` with shape (8, 2, something, 3, 8), where `-1` figures out the size automatically.

```python
a = torch.ones(5)
b = a.numpy()
a = torch.from_numpy(b)
```
This converts tensor `a` (all ones) to a NumPy array `b`, then back to a tensor `a`.

#### Matrix Multiplication

```python
import torch
mat1 = torch.rand(2, 3)
mat2 = torch.rand(3, 3)
res = torch.mm(mat1, mat2)
print(res.size())
```
This multiplies two matrices `mat1` (2,3) and `mat2` (3,3), giving `res` with shape (2,3).

```python
batch1 = torch.rand(10, 3, 4)
batch2 = torch.rand(10, 4, 5)
res = torch.bmm(batch1, batch2)
print(res.size())
```
This does batch matrix multiplication on `batch1` (10,3,4) and `batch2` (10,4,5), resulting in `res` with shape (10,3,5).

#### Computational Graphs

```python
import torch
x = torch.ones(2, 2)
y = torch.ones(2, 1)
w = torch.randn(2, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
p = torch.sigmoid(torch.mm(x, w) + b)
loss = -y * torch.log(p) - (1 - y) * torch.log(1 - p)
cost = loss.mean()
```
This sets up a logistic regression model. `x` is input, `w` is weights, `b` is bias (both need gradients). It calculates predictions `p` with sigmoid, computes cross-entropy loss, and averages it into `cost`.

```python
cost.backward()
print(w.grad)
print(b.grad)
```
This computes gradients of `cost` with respect to `w` and `b` and prints them.

#### Training Procedure

Training involves defining a network, looping over data, processing inputs, computing loss, calculating gradients, and updating weights.

#### Building Neural Networks

```python
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
This defines a simple neural network with two layers: `fc1` (784 inputs to 128 outputs) and `fc2` (128 to 10). The `forward` method applies ReLU activation after `fc1`, then passes through `fc2`.

#### CNN Example for MNIST

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```
This defines a CNN for MNIST: two convolutional layers (`conv1`, `conv2`), followed by three fully connected layers (`fc1`, `fc2`, `fc3`). The `forward` method applies convolutions, ReLU, max pooling, flattens the tensor, and passes it through the dense layers. `num_flat_features` calculates the size for flattening.

#### Data Loading

```python
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
```
This loads the CIFAR-10 dataset, applies transformations (to tensor and normalization), and creates a data loader with batch size 4.

#### Loss Function

```python
criterion = nn.MSELoss()
loss = criterion(output, target)
```
This sets up mean-squared error (MSE) loss and computes it between `output` and `target`.

#### Optimizer

```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
This uses stochastic gradient descent (SGD) with learning rate 0.01, clears gradients, computes gradients from `loss`, and updates the network weights.

#### Full Training Example

```python
net = Net()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
```
This trains the network for 2 epochs. It loops over batches, clears gradients, computes outputs and loss, updates weights, and prints the average loss every 2000 batches.

#### Saving and Loading Models

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))
```
This saves the network’s weights to a file and loads them into a new network.

#### Training on GPU

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
```
This checks for a GPU, moves the network and data to it if available, or uses the CPU otherwise.

---

# Best part of pytorch (Distributed Training), although here only single machine model Parallel is described, Distributed is mainly covered in later weeks

### Scale Training on Multiple GPUs with PyTorch

#### Single Machine Data Parallel

```python
model = Net()
model = torch.nn.DataParallel(model)
```
This wraps the model to split data across multiple GPUs on one machine.

#### Single Machine Model Parallel

```python
class Net(torch.nn.Module):
    def __init__(self, *gpus):
        super(Net, self).__init__()
        self.gpu0 = torch.device(gpus[0])
        self.gpu1 = torch.device(gpus[1])
        self.sub_net1 = torch.nn.Linear(10, 10).to(self.gpu0)
        self.sub_net2 = torch.nn.Linear(10, 5).to(self.gpu1)
    def forward(self, x):
        y = self.sub_net1(x.to(self.gpu0))
        z = self.sub_net2(y.to(self.gpu1))
        return z
model = Net("cuda:0", "cuda:1")
```
This splits a model across two GPUs: `sub_net1` on GPU 0, `sub_net2` on GPU 1. The `forward` method processes input `x` through both parts.

#### Distributed Data Parallel

This uses `torch.nn.parallel.DistributedDataParallel` to split data across multiple machines will be covered in later weeks

---

### Profiling with DLProf and PyTorch Catalyst although i use mlflow now

#### Deep Learning Profiler (DLProf)

DLProf helps analyze performance using CLI and a TensorBoard plugin. It uses Nsight Systems and gives tips via Expert Systems.

#### Automatic Mixed Precision (AMP)

AMP speeds up training with mixed precision, supported by `torch.cuda.amp`.

#### PyTorch Catalyst

Catalyst is a framework to make PyTorch training easier and faster.

```python
runner = dl.SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": train_loader, "valid": valid_loader},
    num_epochs=1,
    logdir="./logs",
    verbose=True
)
```
This runs a training loop with Catalyst’s `SupervisedRunner`. It trains `model` using `criterion` and `optimizer` on training and validation data for 1 epoch, saving logs.

#### Custom Runner

```python
class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))
    def handle_batch(self, batch):
        x, y = batch
        x = x.view(len(x), -1)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.batch_metrics["loss"] = loss
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
runner = CustomRunner()
runner.train(
    loaders={"train": train_loader, "valid": valid_loader},
    model=model, criterion=criterion, optimizer=optimizer,
    num_epochs=1, logdir="./logs", verbose=True
)
```
This defines a custom runner. `predict_batch` makes predictions, `handle_batch` processes each batch (reshapes `x`, computes `logits` and `loss`, updates weights if training). Then it trains the model.

#### Callbacks

```python
runner = dl.SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logdir",
    num_epochs=100,
    callbacks=[
        dl.EarlyStoppingCallback(
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            patience=3,
            min_delta=1e-2
        )
    ]
)
```
This adds an early stopping callback. It stops training if the validation loss doesn’t drop by 0.01 for 3 epochs.
