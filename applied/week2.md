### Week 2 (Introduction to containers, Scheduling and resource management, Deepops Session 1)

Note - Most of the material is from the course material(pdfs) shared by the teachers of NPTEL

Starting with the "Introduction to Containers and IDE" document, it walks us through the machine learning workflow and the challenges that come with it. The workflow involves several steps: collecting and preparing data, designing a model, training and evaluating it on shared hardware, deploying it, and then testing it. However, there’s a big issue—about 40% of businesses say it takes over a month to deploy a machine learning model, which is far too long. The delays come from problems like managing software dependencies, accessing hardware with different frameworks, and handling deployment and version control. For example, one project might need a specific version of Python or CUDA, while another needs a different one, and running both on the same system is tricky. Version control also gets messy when multiple teams are working together. To solve this, the document suggests two options: virtual machines or containers with Docker. It favors containers because Docker lets you package an application with all its dependencies—like libraries and settings—into one portable unit that works the same everywhere.

Then, the document introduces NVIDIA NGC, or NVIDIA GPU Cloud, which is a catalog full of GPU-optimized software for AI, high-performance computing (HPC), and visualization. It offers containers, pre-trained models, Helm charts for Kubernetes, and AI toolkits, making it useful for beginners and experts alike. To use it, you sign up for an account, visit the dashboard, search for something like a TensorFlow container, copy the pull command (for example, `nvcr.io/nvidia/tensorflow:22.01-tf1-py3`), and run it in your terminal to download the container image. This simplifies getting started with GPU-accelerated tools without wrestling with installation headaches.

The section wraps up with Jupyter Notebook, a tool that began as IPython in 2011 and grew into a web-based, open-source platform for interactive data science and computing. It supports over 40 languages, including Python, R, and Julia, and is perfect for things like data cleaning, simulations, exploratory analysis, and machine learning. There’s a demo showing how it works with Docker: you check your Docker images with `docker images`, then create a container named "jupyter_demo" using a command like `sudo NV_GPU=0,1,2,3 nvidia-docker run -it -p 3001:8888 --name="jupyter_demo" nvcr.io/nvidia/tensorflow:20.03-tf2-py3`. This sets up TensorFlow 2.1.0 in a container with Jupyter ready to go, tied to specific GPUs.

Moving on to the "Scheduling and Resource Management" document, it dives into how we manage resources for AI and HPC tasks. It starts by explaining the evolution of HPC systems. Long ago, they used single CPUs, then moved to multiple CPUs, added GPUs for more power, and now use distributed systems with many nodes. A cluster is a group of computers linked together to act as one system, controlled by software. The document lists some AI use cases that need these setups: many users working on many nodes on-premise, many users sharing a single node, cloud bursting for extra capacity, edge or IoT applications, and production inferencing. The big challenge here is resource utilization. Dedicated nodes need a lot of coordination, waste resources, and don’t scale easily, but clusters are more efficient, scalable, and simpler to maintain.

This leads into Slurm, a scheduler designed for HPC clusters. Slurm has a controller called `slurmctld` that keeps track of workloads and resources, while compute nodes run `slurmd` daemons to execute the jobs. There are handy commands to work with it: `sinfo` shows the status of nodes and partitions, `sacct` lists jobs, `squeue` checks job states, and `scancel` stops jobs. When you submit a job, you can tweak it with options like `--nodes` to set the number of nodes, `--mem` for memory, `-n` for the number of tasks, `--gres gpu:#` to request GPUs, or `--exclusive` to reserve a whole node. The demo walks through this: you check node status with `sinfo`, start an interactive job with `srun`, peek at GPU usage with `nvidia-smi`, see running jobs with `squeue`, and cancel a job with `scancel`.

The document then shifts to container orchestration, which is about managing containers with features like setting resource limits, scheduling tasks, balancing loads, checking health, handling faults, and auto-scaling. Kubernetes gets a mention as a tool that manages pods, jobs, and services across on-premise, cloud, or hybrid clusters. The section ends by comparing orchestration tools like Kubernetes with traditional schedulers like Slurm. Kubernetes is container-focused, built for microservices but adapted for AI, and popular in enterprises. Slurm, on the other hand, is HPC-focused, works with bare-metal or containers, and has advanced scheduling features, making it a favorite among researchers.

Finally, the "Deep Ops: Deep Dive into Kubernetes with Deployment of Various AI Based Services Session 1 - Kubernetes" document zooms in on Kubernetes. It explains why Kubernetes is so useful: unlike virtual machines, which include a full operating system, containers are lightweight, holding just the app and its libraries. Kubernetes is a framework for running distributed systems reliably, managing scaling, failover, deployment patterns, service discovery, load balancing, storage orchestration, automated rollouts and rollbacks, efficient resource packing, self-healing, and secret or configuration management.

A Kubernetes cluster has a control plane that runs the show and nodes that do the work. You can kick off a cluster with `minikube start --nodes 3`. The nodes are the workers, each with a kubelet to manage them, a container runtime to run containers, and a kube-proxy for networking. You can check them with `kubectl get nodes` or `docker ps`. The smallest unit in Kubernetes is a pod, which holds one or more containers where your applications live. The demo shows how Kubernetes handles issues: you simulate a pod crash by stopping a node with `docker stop minikube-m03`, check the nodes with `kubectl get nodes` to see one’s not ready, and use `kubectl get pods -o wide` to watch how Kubernetes shifts the workload to keep things running.

Tying it all together, this course gives a solid rundown of tools for accelerated machine learning. Containers with Docker and NVIDIA NGC tackle dependency and deployment woes, making it easy to set up and share environments. Slurm steps in to manage resources efficiently in HPC clusters, keeping everything coordinated. Kubernetes takes it further, orchestrating containers at scale with resilience and flexibility, perfect for big AI projects. Understanding these pieces—containers, scheduling, and orchestration—feels essential for streamlining the journey from building a model to putting it into action.

# Brief summary of the commands used

### **Topic: Docker Commands**
#### **PDF: "Introduction to Containers and IDE"**
- **`docker images`**
  Lists all Docker images on your system.
- **`sudo NV_GPU=0,1,2,3 nvidia-docker run -it -p 3001:8888 --name="jupyter_demo" nvcr.io/nvidia/tensorflow:20.03-tf2-py3`**
  Launches a container named "jupyter_demo" from a TensorFlow image, mapping ports and assigning GPUs.

---

### **Topic: Slurm Commands**
#### **PDF: "Scheduling and Resource Management"**
- **`sinfo`**
  Displays node and partition status in the cluster.
- **`srun --ntasks=5 --nodes=1 --cpus-per-task=2 --partition=batch --time=4:00:00 --gres=gpu:1 --pty /bin/bash`**
  Requests an interactive job with specific resources (tasks, CPUs, GPU, time).
- **`nvidia-smi`**
  Shows GPU usage details.
- **`squeue`**
  Lists all active or queued jobs.
- **`scancel <Job Id>`**
  Stops a job using its ID.

---

### **Topic: Kubernetes Commands**
#### **PDF: "Deep Ops: Deep Dive into Kubernetes with Deployment of Various AI Based Services Session 1 - Kubernetes"**
- **`minikube start --nodes 3`**
  Starts a local Kubernetes cluster with 3 nodes.
- **`kubectl get nodes`**
  Lists all nodes in the Kubernetes cluster.
- **`docker stop minikube-m03`**
  Stops a specific node in the cluster.
- **`kubectl get pods -o wide`**
  Displays detailed pod information, including node assignments.

---
