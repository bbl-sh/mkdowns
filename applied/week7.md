# Week 7 (Distributed Deep learning)

Note - Most of the material is from the course material(pdfs) shared by the teachers of NPTEL


let's go(chama), begin with the need of distributed deep learning, Deep learning models are growing larger and require more computing power than a single computer can provide. For example, in 2015, a model had 60 million parameters and needed significant computation to train. By 2017, another model had 8,700 million parameters, requiring even more power. To handle this, we use many computers or GPUs to split the work.

---

### Motivations for Multi-GPU Training
Using multiple GPUs allows us to train models on larger datasets faster. This improves performance for tasks like identifying objects in images. More data leads to better models, and multiple GPUs make this possible by speeding up the process.

---

### The Scaling Laws
There’s a formula: *L = (N / 8.8 × 10¹³)^(-0.076)*. It shows that larger models can reduce errors, but there’s a limit to how much improvement you get as the model size increases.

---

### Types of Parallelism
There are two main ways to divide the work in deep learning:

- **Data Parallelism**:
  - The training data is split across multiple GPUs.
  - Each GPU has the same copy of the model but works on a different portion of the data.
  - They share updates to the model weights to improve it together.

- **Model Parallelism**:
  - The model itself is too large to fit on one GPU.
  - Different parts of the model are placed on different GPUs.
  - These GPUs work together to process the data.

---

### Synchronous vs Asynchronous Training
There are two ways GPUs or workers can coordinate during training:

- **Synchronous Training**:
  - All GPUs or workers train at the same time.
  - They wait for everyone to finish their work before updating the model.
  - This keeps everything consistent but can be slower if some GPUs take longer.

- **Asynchronous Training**:
  - Each GPU or worker trains independently.
  - They update the model as soon as they finish, without waiting for others.
  - This can be faster but might lead to less consistent updates.

---

### NVIDIA's Approach: NCCL
NVIDIA’s NCCL (NVIDIA Collective Communications Library) is a tool that helps multiple GPUs and computers communicate quickly while using fewer resources. It’s designed to make distributed training efficient.

---

### System Topology and NCCL
System topology refers to how computers, GPUs, network cards, and switches are connected. NCCL analyzes this setup to find the fastest way to send data between GPUs. It works across multiple computers (called nodes) and uses fast technologies like:

- **NVLink**: A direct, high-speed connection between GPUs.
- **RDMA (Remote Direct Memory Access)**: A way to move data over networks quickly.

This ensures efficient communication, whether GPUs are on the same computer or spread across many nodes.

---

### Framework Support
Popular deep learning frameworks like **TensorFlow** and **PyTorch** can use NCCL to perform distributed training across multiple GPUs or computers.

---

### NVIDIA NGC Support
NVIDIA’s NGC (NVIDIA GPU Cloud) provides pre-built packages (called containers) that simplify setting up distributed deep learning on multiple computers. These containers come with multi-node support already included.

---

### Introduction to TensorFlow Distributed
TensorFlow uses the `tf.distribute` API to train models across multiple GPUs, computers, or TPUs. It requires only small changes to your code to work.

---

### TensorFlow's tf.distribute API
TensorFlow offers several strategies for distributing training:

- **MirroredStrategy**:
  - Used for multiple GPUs on one computer.
  - Copies the model to each GPU and keeps them synchronized.

- **MultiWorkerMirroredStrategy**:
  - For multiple computers, each with GPUs.
  - Works like MirroredStrategy but across machines.

- **CentralStorageStrategy**:
  - Stores model variables on the CPU and copies operations to all GPUs.

- **ParameterServerStrategy**:
  - Uses separate servers to store variables.
  - Workers read from and update these servers.

- **OneDeviceStrategy**:
  - Runs everything on a single device, like one GPU.

---

### Introduction to Horovod
Horovod is a tool that simplifies distributed deep learning for frameworks like TensorFlow, Keras, PyTorch, and MXNet. It’s easy to use and works well for large-scale tasks.

---

### Horovod with TensorFlow
To use Horovod with TensorFlow, follow these steps:

1. **Initialize Horovod**: Start Horovod to prepare for distributed training.
2. **Pin GPUs**: Assign each GPU to a specific process.
3. **Wrap the Optimizer**: Use Horovod’s optimizer, which averages updates from all GPUs.
4. **Broadcast Variables**: Share the initial model values from one process to all others.
5. **Save Checkpoints Carefully**: Ensure only one process saves progress to avoid errors.

---
# Code in week 7

### Running Horovod
Horovod can run on one computer or many. Examples:

- **One computer with 4 GPUs**:
  ```bash
  horovodrun -np 4 python train.py
  ```
- **Two computers, each with 4 GPUs**:
  ```bash
  horovodrun -np 8 -H server1:4,server2:4 python train.py
  ```
---

### Data Parallelism Demo
In **data parallelism**, the same model is replicated across multiple GPUs, and each GPU processes a different portion of the input data. The gradients are then averaged across GPUs to update the model. PyTorch simplifies this with `torch.nn.DataParallel`.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)

# Enable data parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
inputs = torch.randn(100, 10).to(device)
targets = torch.randn(100, 1).to(device)

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

**What’s Happening**:
- The model is a simple linear layer (`10 inputs → 1 output`).
- `DataParallel` splits the input data (`inputs`) across available GPUs, computes the forward and backward passes in parallel, and averages the gradients.
- If no GPUs are available, it runs on the CPU without parallelism.

---

### Model Parallelism Demo
In **model parallelism**, different parts of the model are placed on separate GPUs. This is useful for large models that don’t fit on a single GPU. Here, we split a model into two parts across two GPUs.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# First part of the model
class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.fc1 = nn.Linear(10, 100)

    def forward(self, x):
        return self.fc1(x)

# Second part of the model
class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        return self.fc2(x)

# Setup devices and model parts
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")
part1 = ModelPart1().to(device1)
part2 = ModelPart2().to(device2)

# Loss and optimizer for both parts
criterion = nn.MSELoss()
optimizer = optim.SGD(list(part1.parameters()) + list(part2.parameters()), lr=0.01)

# Dummy data
inputs = torch.randn(100, 10).to(device1)
targets = torch.randn(100, 1).to(device2)

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    # Forward pass: part1 on GPU 0, part2 on GPU 1
    intermediate = part1(inputs)
    outputs = part2(intermediate.to(device2))
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

**What’s Happening**:
- The model is split into two parts: `part1` (on `cuda:0`) transforms `10 inputs → 100 outputs`, and `part2` (on `cuda:1`) transforms `100 inputs → 1 output`.
- The intermediate result from `part1` is moved to `cuda:1` for `part2` to process.
- Gradients are computed and backpropagated across both parts.

---

### Notes
- **Data Parallelism**: Ideal for speeding up training with larger batches across GPUs. Requires multiple GPUs to see the effect.
- **Model Parallelism**: Useful for large models, but requires manual device management and data movement between GPUs.
- These demos assume at least two GPUs (`cuda:0` and `cuda:1`). If you don’t have multiple GPUs, the model parallelism example will raise an error unless modified to use a single device.
