# Week 8 (Convergence, Fundamentals of accelerating deployment)

## Fundamentals of accelerating deployment

Note - Most of the material is from the course material(pdfs) shared by the teachers of NPTEL

### Key Topics

- **Training and Inference**:
  - **Training**: Teaching a neural network using existing data to create a trained model.
  - **Inference**: Using the trained model to make predictions on new data.
  - Deployment is about using the model in real-world systems to make decisions.

- **Deployed ML Inference Systems**:
  - Deployment can happen on the cloud or edge devices (like phones or IoT devices).
  - Performance is measured by:
    - **Accuracy**: How correct the predictions are.
    - **Latency**: Time taken to get a prediction.
    - **Throughput**: Number of predictions per second.
    - **Energy/Power**: Energy used per prediction or by the system.
    - **Model Size**: Storage space needed for the model.
    - **Memory Use**: Memory required during operation.
    - **Cost**: Money spent on deployment.
  - There are trade-offs between these metrics (e.g., improving speed might reduce accuracy).

- **Model Deployment**:
  - Means integrating the model into a production environment to process inputs and provide outputs for business decisions.
  - Important for:
    - Reliable predictions for other software.
    - Maximizing the model's value.

- **Making Models Smaller**:
  - Goal: Reduce model size and computation needs while keeping accuracy.
  - Techniques:
    1. **Low-Precision Inference**: Use simpler math (e.g., INT8 instead of FP32) for faster inference.
    2. **Pruning**: Remove weights or activations near zero to shrink the model, often with fine-tuning.
    3. **Old-School Compression**: Apply traditional methods like Huffman coding to compress weights.
    4. **Knowledge Distillation**: Train a small "student" model to copy a large "teacher" model.
    5. **Efficient Architectures**: Use designs like MobileNet or ShuffleNet, built for efficiency.

- **Low-Precision Inference**:
  - Converts FP32 (32-bit floating point) weights and activations to INT8 (8-bit integer).
  - Benefits:
    - Less storage and bandwidth needed.
    - Faster and more energy-efficient than FP32.
  - INT8 saves 30x energy and 116x area compared to FP32 for additions.
  - Requires calibration to maintain accuracy, which is automatic.

- **NVIDIA TensorRT**:
  - A tool to optimize deep learning models for inference on NVIDIA GPUs.
  - Offers low latency and high throughput.
  - Capabilities:
    - Supports C++ and Python APIs.
    - Has build and runtime phases.
    - Includes plugins, quantization, dynamic shapes, and more.

- **Features of TensorRT**:
  1. **Reduced Precision**: Uses FP16 or INT8 to speed up inference while keeping accuracy.
  2. **Layer and Tensor Fusion**: Combines layers (e.g., convolution, bias, ReLU) to reduce computations and GPU memory use.
  3. **Auto-Tuning Kernel**: Picks the best algorithms for the GPU and input data.
  4. **Dynamic Tensor Memory**: Allocates memory only when needed, reducing waste.
  5. **Multi-Stream Execution**: Processes multiple inputs at once.
  6. **Time Fusion**: Optimizes recurrent neural networks over time.

- **Supported Layers**:
  - Convolution (2D), activation (ReLU, tanh, sigmoid), pooling (max, average), element-wise operations (sum, product, max), LRN (cross-channel), fully connected, softmax, deconvolution.

- **Deploying with TensorRT**:
  - Creates an optimized runtime inference engine that can be saved (serialized) to disk for later use.

---

## Challenges with Convergence

### Key Topics

- **Impact of Batch Size**:
  - Batch size affects training and accuracy.
  - Larger batch sizes may speed up training but can increase validation error (poorer generalization).
  - Figure 1 (not visible) likely shows how training and validation errors change with batch size.

- **Flat vs. Sharp Minima**:
  - The loss function has minima (low points) that can be flat or sharp.
  - Flat minima are better for generalization because they are less sensitive to parameter changes.
  - Figure 1 (not visible) likely illustrates this concept.

- **Techniques for Faster Convergence**:
  1. **Ghost Batch Normalization**:
     - A tweak to batch normalization (BN) that uses statistics from smaller "ghost" batches instead of the full batch.
     - Helps reduce generalization error with large batch sizes.
     - Full batch statistics are still used during inference.
     - Referenced in Hoffer et al. (2017).
  2. **Learning Rate Scaling**:
     - Rule: If batch size increases by k (e.g., 2x), increase learning rate by k (e.g., 2x).
     - Keeps training stable and matches accuracy and training curves for small and large batches.
     - Useful for distributed learning with multiple workers.
  3. **Learning Rate Warmup**:
     - Starts with a low learning rate and gradually increases it early in training.
     - Prevents instability, especially with large batches.
     - A graph (not visible) likely shows this over epochs.
  4. **Optimizers for Exascale Deep Learning**:
     - Special optimizers designed for very large-scale training (e.g., batch sizes of 16k or 32k).
     - Referenced in You et al. (2017) for fast ImageNet training.

---
# Code in week 8 (Deployment using TensorRT ofcourse using pytorch)

This script demonstrates how to take a pre-trained PyTorch model, convert it to ONNX format, optimize it with TensorRT, and perform inference. The code assumes you have an NVIDIA GPU, along with PyTorch, TensorRT, NumPy, and PyCUDA installed.
```python
import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Load pre-trained model in PyTorch
model = models.resnet18(pretrained=True)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")

# Build TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("resnet18.onnx", "rb") as model_file:
    parser.parse(model_file.read())
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1 GB
engine = builder.build_engine(network, config)
with open("resnet18_engine.trt", "wb") as f:
    f.write(engine.serialize())

# Load the engine for inference
with open("resnet18_engine.trt", "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Prepare input and output buffers
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
h_input = input_data
h_output = np.empty((1, 1000), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
cuda.memcpy_htod(d_input, h_input)

# Run inference
bindings = [int(d_input), int(d_output)]
context.execute_v2(bindings)
cuda.memcpy_dtoh(h_output, d_output)

# Get predicted class
predicted_class = np.argmax(h_output)
print(f"Predicted class: {predicted_class}")
```
