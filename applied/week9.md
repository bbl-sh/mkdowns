# Week 9 (Accelerating Deployments with TensorRT, RAPIDS)

Note - Most of the material is from the course material(pdfs) shared by the teachers of NPTEL

# Accelerating Deployments with TensorRT

This document covers the fundamentals of accelerating deployments using TensorRT, including notes and code examples from a PDF.

## Fundamentals of Accelerating Deployments

## Deploying a Model with TensorRT

- The output of TensorRT optimization is a runtime inference engine that can be serialized for distribution.

## Supported Layer Types in TensorRT

TensorRT supports these layer types:

- Convolution: 2D
- Activation: ReLU, tanh, and sigmoid
- Pooling: max and average
- ElementWise: sum, product, or max of two tensors
- LRN: cross-channel only
- Fully-connected: with or without bias
- SoftMax: cross-channel only
- Deconvolution

## Steps for TensorRT Application

1. Convert the pretrained image segmentation PyTorch model into ONNX.
2. Import the ONNX model into TensorRT.
3. Apply optimizations and generate an engine.
4. Perform inference on the GPU.

## Components in TensorRT

- **ONNX Parser**: Takes a trained model in ONNX format and populates a network object in TensorRT.
- **Builder**: Takes a TensorRT network and generates an optimized engine for the target platform.
- **Engine**: Takes input data, performs inferences, and produces inference output.
- **Logger**: Captures errors, warnings, and other information during the build and inference phases.

## Performance Metrics

- TensorRT-powered NVIDIA wins all performance tests in the MLPerf Inference benchmark.
- It accelerates models in computer vision, speech-to-text, natural language understanding (BERT), and recommender systems.
- Source: https://developer.nvidia.com/tensort

## Framework Integration

- TensorRT supports all major frameworks like PyTorch and TensorFlow.
- It can achieve 6x faster inference with 1 line of code.

## TensorFlow-TensorRT (TF-TRT)

- TF-TRT optimizes and executes compatible subgraphs in TensorFlow models.
- TensorRT parses the model and applies optimizations where possible.
- It sped up TensorFlow inference by 8x for low-latency runs of the ResNet-50 benchmark.
- Note: Graph optimizations do not change the underlying computation; they restructure it for faster, more efficient performance.
- Source: https://developer.nvidia.com/blog/tensorrt-integration-speeds-tensorflow-inference/

### Benefits of Integrating TensorFlow with TensorRT

- TensorRT optimizes the largest subgraphs possible in the TensorFlow graph.
- More compute in the subgraph means greater performance benefits.
- The goal is to optimize most of the graph with the fewest TensorRT nodes for best performance.
- Depending on operations, the final graph might have more than one TensorRT node.

### Workflow of TF-TRT

1. Create and train an ML model or get a pretrained ML model.
2. Save the model as a TensorFlow SavedModel.
3. Optimize it using TF-TRT to create a TRT-optimized TensorFlow SavedModel.
4. Deploy with TensorFlow or TRITON Inference Server.

### Code Snippets for TF-TRT in TensorFlow 1.x

- **SavedModel Format**:
  ```python
  from tensorflow.python.compiler.tensorrt import trt_convert as trt
  ```

- Note: You need to create a SavedModel (or frozen graph) from a trained TensorFlow model and pass it to the TF-TRT Python API. It returns an optimized SavedModel (or frozen graph) by replacing supported subgraphs with TRTEngineOp nodes.

## Installing TF-TRT

- NVIDIA containers of TensorFlow include TensorRT by default.
- TF-TRT is part of the TensorFlow binary in the container and can be used out of the box with all required software dependencies.

## Conversion Parameters in TF-TRT

Parameters for `saved_model_cli` and `TrtGraphConverter`:

- **precision_mode**: FP32, FP16, or INT8
- **minimum_segment_size**: Minimum number of TensorFlow nodes required for a TensorRT subgraph.
- **is_dynamic_op**: Builds TensorRT engines at runtime (for dynamic shapes).
- **use_calibration**: For INT8 mode, creates a calibration graph if True (recommended).
- **max_batch_size**: Maximum batch size for TensorRT engines (used when `is_dynamic_op=False`).
- **maximum_cached_engines**: Limits cached TensorRT engines per TRTEngineOp (used when `is_dynamic_op=True`).

## Benchmarking CPU, GPU (CUDA), and TensorRT Performance (PyTorch Demo)

### Step 1: Load a Pretrained Neural Network

```python
import torch
from torchvision import models, transforms

# Using CPU
model = models.resnet50(pretrained=True).to("cuda")
```

### Step 2: Load an Example Image

```python
from PIL import Image
img = Image.open("img1.jpg")
```

### Step 3: Transform Example Image (Preprocessing)

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
img = transform(img)
print(img.shape)  # torch.Size([3, 224, 224])
```

### Step 4: Set a Batch Size

```python
# Using GPU
img_batch = torch.unsqueeze(img, 0).to("cuda")
print(img_batch.shape)  # torch.Size([1, 3, 224, 224])
```

### Step 5: Make a Prediction

```python
model.eval()
with torch.no_grad():
    outputs = model(img_batch)
prob = torch.nn.functional.softmax(outputs[0], dim=0)
```

### Step 6: Extract Top 5 Probabilities

```python
import pandas as pd
categories = pd.read_csv('https://raw.githubusercontent.com')
topk = 5
probs, classes = torch.topk(prob, topk)
for i in range(topk):
    probability = probs[i].item()
    class_label = categories[0][int(classes[i])]
    print(f"{int(probability*100)}% {class_label}")
# Output:
# 37% Egyptian cat
# 19% tabby
# 11% tiger cat
# 4% Siamese cat
# 3% cartoon
```

### Step 7: Define a Benchmark Function

```python
import time
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def benchmark(model, device="cuda", input_shape=(32, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape).to(device)
    print("Warm up...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(f'Iteration {i}/{nruns}, ave batch time {np.mean(timings) * 1000:.2f} ms')
    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print(f"Average batch time: {np.mean(timings) * 1000:.2f} ms")
```

### Step 8: CPU Benchmark

```python
benchmark(model, device="cpu")
# Output:
# Warm up...
# Start timing...
# Iteration 10/100, ave batch time 640.36 ms
# Iteration 20/100, ave batch time 642.35 ms
# ...
# Average batch time: 645.74 ms
```

### Step 9: CUDA Benchmark

```python
model = model.to("cuda")
benchmark(model)
# Output:
# Warm up...
# Start timing...
# Iteration 10/100, ave batch time 92.37 ms
# Iteration 20/100, ave batch time 92.42 ms
# ...
# Average batch time: 106.73 ms
```

### Step 10: Compile to TensorRT and Benchmark

```python
traced_model = torch.jit.trace(model, [torch.randn((32, 3, 224, 224)).to("cuda")])
import torch_tensorrt
trt_model = torch_tensorrt.compile(traced_model,
    inputs=[torch_tensorrt.Input((32, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions={torch.float32}
)
benchmark(trt_model)
# Output:
# Warm up...
# Start timing...
# Iteration 10/100, ave batch time 76.66 ms
# Iteration 20/100, ave batch time 75.95 ms
# ...
# Average batch time: 71.52 ms
```

## Demo of Workflow for Optimizing TensorFlow Model to TensorRT

### Task 1: Convert TensorFlow Model to Frozen Model (*.pb)

```python
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
    your_outputs = ["output_tensor/Softmax"]
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=your_outputs
    )
    with gfile.FastGFile("./model/frozen_model.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")
```

### Task 2: Optimize Frozen Model to TensorRT Graph

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=your_outputs,
    max_batch_size=2,
    max_workspace_size_bytes=2 * (10**9),
    precision_mode="INT8"
)
with gfile.FastGFile("./model/TensorRT_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")
```

### Task 3: Count Nodes Before and After Optimization

- Optimization log:
  - Original graph: 36 nodes reduced to 13 nodes after TensorRT optimization.
  - Time for TensorRT optimization: 5030.22607 ms.

---

# RAPIDS: The Platform Inside and Out

overview of RAPIDS, an open-source platform that speeds up data science tasks like data preparation, model training, and visualization using GPUs. It includes notes and code examples from the OCR document, covering key components such as cuDF, cuML, and cuGraph, along with their technology stacks, algorithms, and performance details.

## Open Source Data Science Ecosystem with Familiar Python APIs

RAPIDS is an open-source system that works with common Python tools to make data science faster using GPUs. It supports:

- **Data Preparation**: Getting data ready for analysis.
- **Model Training**: Building machine learning models.
- **Visualization**: Creating visual insights from data.

It repeats these core areas multiple times, emphasizing their importance in the RAPIDS ecosystem.

## cuDF: GPU-Accelerated DataFrames

cuDF is a key part of RAPIDS that handles data tables (DataFrames) on GPUs, similar to the Pandas library in Python but faster.

### ETL Technology Stack

The ETL process (Extract, Transform, Load) is essential for preparing data. RAPIDS speeds this up with the following tools:

- **Python**: For writing scripts.
- **Cython**: For making Python code run faster (listed as "CUTION" in the OCR, likely a typo).
- **CUDA**: The GPU computing platform.

### libcuDF: CUDA C++ Library

libcuDF is the C++ foundation of cuDF, offering:

- DataFrame and column types with algorithms.
- CUDA operations for tasks like sorting, joining, grouping, and more.
- Optimized GPU support for strings, timestamps, and numbers.
- Tools for large-scale data processing.

**Example Code**:
```cpp
std::unique_ptr<table> gather(table_view const& input, column_view const& gather_map, ...)
// Returns a new table with rows from "input" selected by "gather_map"
```

### cuDF: Python Library

cuDF also provides a Python version that mimics Pandas for GPU data handling. It includes:

- Creating GPU DataFrames from NumPy arrays, Pandas DataFrames, and PyArrow Tables.
- Fast compilation of custom functions using Numba.

**Example Code**:
```python
# Reading data with cuDF (incomplete in OCR, corrected for clarity)
gdf = cudf.read_csv('path/to/data.csv')
# Note: It decompresses data as it loads into memory
```

## Machine Learning with cuML

cuML is the RAPIDS library for machine learning, offering GPU versions of Scikit-Learn algorithms.

### ML Technology Stack

The machine learning tools in RAPIDS include:

- **Python**: For writing code.
- **Cython**: For speeding up Python.
- **cuML Algorithms**: Machine learning models on GPUs.
- **cuML Prims**: Basic building blocks for machine learning.
- **CUDA Libraries**: For GPU operations.
- **CUDA**: The GPU platform.

### GPU-Accelerated Clustering Example

RAPIDS makes clustering faster on GPUs with simple code changes from CPU versions.

**CPU-based Clustering (Scikit-Learn)**:
```python
from sklearn.datasets import make_moons
import pandas

X, y = make_moons(n_samples=int(1e2), noise=0.05, random_state=0)
X = pandas.DataFrame({'feald' + str(i): X[:, i] for i in range(X.shape[1])})

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_hat = dbscan.fit_predict(X)
```

**GPU-Accelerated Clustering (cuML)**:
```python
from sklearn.datasets import make_moons
import cudf

X, y = make_moons(n_samples=int(1e2), noise=0.55, random_state=0)
X = cudf.DataFrame({f'facility': ...})  # OCR incomplete, likely a column definition error

from cuml import DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_hat = dbscan.fit_predict(X)
```

### Algorithms in cuML

cuML includes these GPU-accelerated algorithms:

- Decision Trees / Random Forests
- Linear/Lasso/Ridge/LARS/ElasticNet Regression
- Logistic Regression
- K-Nearest Neighbors (exact or approximate)
- Support Vector Machine (SVM) Classification and Regression
- Naive Bayes
- Random Forest / Gradient Boosted Decision Trees (GBDT) Inference (FIL)
- Text Vectorization (TF-IDF / Count)
- Target Encoding
- Cross-validation / Splitting
- K-Means
- DBSCAN
- Spectral Clustering
- Principal Components (including iPCA)
- Singular Value Decomposition (SVD)
- UMAP
- Spectral Embedding
- T-SNE
- Holt-Winters
- Seasonal ARIMA / Auto ARIMA

The document notes "More to come!" repeatedly, suggesting ongoing development.

### Forest Inference Library (FIL)

cuML’s FIL speeds up predictions for random forests and boosted decision trees:

- Works with models from XGBoost, LightGBM, Scikit-Learn RF (cuML RF coming soon).
- Simple Python API.
- A single V100 GPU is up to 34x faster than a dual-CPU node for XGBoost.

**Performance Example**:
- For 1000 trees, FIL on a V100 GPU beats XGBoost on a 40-core CPU.

## Integration with Cloud ML Frameworks

RAPIDS works with cloud platforms to speed up machine learning and hyperparameter optimization (HPO):

- Trains models up to 25x faster than CPUs.
- Integrates with Amazon SageMaker, Azure ML, Google AI Platform, and open-source tools like Dask, Ray, and Tune.
- Offers code samples at `https://rapids.ai/hpo`.

**HPO Use Case**:
- A 100-job Random Forest model for airline data showed over 7x reduction in total cost of ownership (TCO).

## cuGraph: GPU-Accelerated Graph Analytics

cuGraph speeds up graph analytics, helping find insights from connected data.

### Graph Technology Stack

The graph analytics tools include:

- **Python**: For scripting (repeated multiple times in OCR).
- **CUDA**: For GPU computing (implied but not fully listed).

### Algorithms in cuGraph

cuGraph supports these GPU-accelerated graph algorithms:

- **Community**: Spectral Clustering, Balanced-Cut, Modularity Maximization, Louvain, Subgraph Extraction.
- **Components**: Triangle Counting, Weakly/Strongly Connected Components.
- **Link Analysis**: PageRank (Multi-GPU), Personal PageRank.
- **Link Prediction**: Jaccard, Weighted Jaccard, Overlap Coefficient.
- **Traversal**: Single Source Shortest Path (SSSP), Breadth First Search (BFS).
- **Structure**: COO-to-CSR (Multi-GPU), Transpose, Renumbering.

### Performance Example: Louvain Algorithm

cuGraph’s Louvain algorithm outperforms NetworkX for community detection.

**Code Example**:
```python
import cugraph
G = cugraph.Graph()
G.add_edge_list(gdf['src_0'], gdf['dst_0'], gdf['data'])
df, mod = cugraph.nvLouvain(G)
# Returns a DataFrame with "vertex" (ID) and "partition" (assigned group)
```

**Performance Data**:
| Graph                  | Nodes    | Edges      |
|-----------------------|----------|------------|
| preferentialAttachment| 100,000  | 999,970    |
| caidaRouterLevel      | 192,244  | 1,218,132  |
| coAuthorsDBLP         | 299,067  | 299,067    |
| dblp-2010             | 326,186  | 1,615,400  |
| citationCiteseer      | 268,495  | 2,313,294  |
| coPapersDBLP          | 540,486  | 30,491,458 |
| coPapersCiteseer      | 434,102  | 32,073,440 |
| as-Skitter            | 1,696,415| 22,190,596 |

- Speedup reaches up to 12,000x compared to NetworkX.

### Goals and Benefits of cuGraph

- **Performance**: Handles 500 million edges on a 32GB GPU, scales to billions with multi-GPU.
- **Integration**: Works with cuDF and cuML using DataFrames.
- **APIs**: Python (NetworkX-like) and C/C++ for detailed control.
- **Functionality**: Offers a growing set of algorithms and tools.
