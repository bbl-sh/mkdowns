# week 3 (Deepops_Session_II, Networking_Fundamentals)

## Deepops_Session_II

Note - Most of the material is from the course material(pdfs) shared by the teachers of NPTEL

Let's begin with Kubernetes as is a helpful tool for managing containers and keeping applications running smoothly. It excels in sharing nodes and scheduling GPU jobs, especially in cloud environments. Kubernetes also takes care of data permissions, security, analytics, and monitoring. On the other hand, SLURM is specialized for High-Performance Computing (HPC). It efficiently manages resources on bare-metal hardware and is great for tasks involving multiple nodes, advanced scheduling, job dependencies, and HPC-specific functionality.

DevOps practices enable software to quickly move from development to production while ensuring reliability. These practices use automation and workflows to simplify developers’ work. MLOps builds upon DevOps principles and specifically supports Machine Learning (ML) by bridging the gap between continuously evolving code and real-world data. It relies on structured data pipelines, forming ML pipelines for training models and serving predictions. Continuous integration and deployment (CI/CD) processes streamline the overall ML workflow.

## Networking_Fundamentals

Networks interconnect computing elements such as CPUs and GPUs, playing a critical role in High-Performance Computing (HPC). Traditional SMP (Symmetric Multiprocessing) machines differ significantly from modern clusters, which use network switches, adapters, and cables like Ethernet and InfiniBand to enhance speed and efficiency.

Data transmission methods include NRZ (Non-Return to Zero), which transmits one bit per clock and is considered traditional, and PAM4 (Pulse Amplitude Modulation), an advanced method doubling the data rate suitable for high-speed connections. Eye diagrams visually represent signal quality, showing a clear "eye" for good signals and a closed eye indicating degraded signals.

Optical components are essential due to signal degradation (insertion loss) over distances. Technologies such as VCSEL lasers convert electrical signals to optical signals and back, improving transmission quality, especially with high-speed hardware like NVIDIA’s HDR InfiniBand switches.

Remote Direct Memory Access (RDMA) is a significant advancement, enabling direct memory transfers between systems without involving the CPU, thus reducing latency. GPUDirect RDMA further enhances efficiency by allowing GPUs direct network data access without extra memory copies, significantly lowering latency and CPU load.

Networks use various organizational structures (topologies) like Fat Tree, Torus, and Dragonfly. Adaptive routing ensures efficient data paths, minimizing congestion and maximizing bandwidth. Technologies such as SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) enable networks to perform data aggregation and computation directly, greatly benefiting AI and ML tasks by significantly speeding up processing.

Nvidia’s GPUs, particularly the A100 models, provide robust performance with varied memory configurations and connectivity technologies like NVLink. DGX systems, such as the DGX A100, combine multiple GPUs and CPUs optimized for intensive AI workloads, equipped with rapid networking and advanced memory solutions.

Modern networks are resilient and can self-heal by detecting faults or congestion and automatically rerouting data to maintain consistent performance. Latency in networks results from components like cables, switches, and adapters, and each adds incremental delays affecting overall speed. Understanding and minimizing these latencies is essential for achieving optimal performance in HPC environments.

## There was lot's of code in this week and so i will be implementing projects which will include most of the topics
Below is a detailed explanation of commands related to SLURM, Kubernetes, and DeepOps, formatted in markdown with code blocks and notes-like explanations. These commands are commonly used for managing compute resources in high-performance computing (HPC) and AI workloads. I’ve included all the commands with explanations as requested, ensuring the code remains formatted correctly in markdown code blocks (```bash) and providing clear reasoning for each.

---
# Summary of the commands used in week 3

## SLURM Commands

SLURM (Simple Linux Utility for Resource Management) is a job scheduler for HPC clusters. It helps users submit, monitor, and manage jobs across multiple nodes. Below are the key SLURM commands with explanations:

- **`srun`**
  ```bash
  srun --nodes=1 --ntasks=1 hostname
  ```
  about: This command runs a job interactively or launches a single task. Here, it executes the `hostname` command on one node with one task. It’s great for quick tests or starting parallel processes in real-time.

- **`sbatch`**
  ```bash
  sbatch my_script.sh
  ```
  about: Submits a batch script (e.g., `my_script.sh`) to SLURM for non-interactive execution. The script can include multiple commands, making it ideal for long-running jobs like simulations or training AI models.

- **`squeue`**
  ```bash
  squeue
  ```
  about: Lists all jobs in the queue, showing their status (e.g., running, pending), job IDs, and assigned nodes. It’s your go-to command to check what’s happening in the cluster.

- **`sinfo`**
  ```bash
  sinfo
  ```
  about: Displays info about nodes and partitions (queues) in the cluster, like which nodes are free or busy. Useful for planning jobs or troubleshooting resource availability.

- **`sacct`**
  ```bash
  sacct --jobs=12345
  ```
  about: Shows accounting details for a job (e.g., CPU time, memory used) after it runs. Replace `12345` with your job ID. It’s handy for analyzing resource usage post-job.

- **`scancel`**
  ```bash
  scancel 12345
  ```
  about: Cancels a job by its ID (e.g., `12345`). Use this to stop a running or pending job if something goes wrong or priorities change.

- **`scontrol`**
  ```bash
  scontrol show node=node01
  ```
  about: Lets you view or tweak SLURM settings. This example shows details about `node01` (e.g., its state or resources). It’s more for admins but useful for debugging.

- **`salloc`**
  ```bash
  salloc --nodes=2 --time=01:00:00
  ```
  about: Allocates resources (e.g., 2 nodes for 1 hour) for an interactive session. Perfect when you need compute power right away, like for testing code.

- **`sattach`**
  ```bash
  sattach 12345.0
  ```
  about: Connects to a running job (ID `12345`, step `0`) to see its output or interact with it. Think of it as jumping into a job mid-run.

- **`sbcast`**
  ```bash
  sbcast input.txt /tmp/input.txt
  ```
  about: Copies a file (`input.txt`) to all nodes assigned to your job (to `/tmp/input.txt`). Useful for distributing data or scripts needed by your job.

- **`sshare`**
  ```bash
  sshare
  ```
  about: Shows fairshare info, which balances resource use among users. It helps ensure no one hogs the cluster.

- **`sprio`**
  ```bash
  sprio -j 12345
  ```
  about: Displays priority factors (e.g., job age, size) for a job (ID `12345`). Helps you understand why some jobs run before others.

- **`sstat`**
  ```bash
  sstat --jobs=12345
  ```
  about: Gives real-time stats (e.g., CPU, memory use) for a running job (ID `12345`). Great for monitoring performance live.

- **`strigger`**
  ```bash
  strigger --set --jobid=12345 --flags=completion
  ```
  about: Sets a trigger to run an action (e.g., notify you) when a job (ID `12345`) completes. It’s for automating follow-ups.

- **`sview`**
  ```bash
  sview
  ```
  about: Opens a graphical interface to manage the cluster. It’s a visual alternative to command-line tools, showing nodes, jobs, etc.

---

## Kubernetes Commands

Kubernetes manages containerized applications, and `kubectl` is its command-line tool. These commands are key for creating and monitoring resources like pods (containers) in a cluster, especially for AI services.

- **`kubectl get <resource>`**
  ```bash
  kubectl get pods
  ```
  about: Lists resources (e.g., pods) in the cluster. This checks what’s running, like ML pods or services, in your current namespace.

- **`kubectl describe <resource> <name>`**
  ```bash
  kubectl describe pod my-pod
  ```
  about: Shows detailed info about a resource (e.g., pod `my-pod`), like its settings or errors. Perfect for troubleshooting.

- **`kubectl create -f <filename>`**
  ```bash
  kubectl create -f pod.yaml
  ```
  about: Creates a resource from a YAML file (e.g., `pod.yaml`). Use this to set up a new pod or service with a predefined config.

- **`kubectl apply -f <filename>`**
  ```bash
  kubectl apply -f ml-pod.yaml
  ```
  about: Applies or updates a resource from a file (e.g., `ml-pod.yaml`). It’s flexible for creating or modifying ML pods—common in demos.

- **`kubectl delete <resource> <name>`**
  ```bash
  kubectl delete pod my-pod
  ```
  about: Deletes a resource (e.g., pod `my-pod`). Use this to clean up when a pod’s no longer needed.

- **`kubectl logs <pod-name>`**
  ```bash
  kubectl logs ml-pod
  ```
  about: Fetches logs from a pod (e.g., `ml-pod`). Essential for checking output or debugging an ML application.

- **`kubectl exec -it <pod-name> -- <command>`**
  ```bash
  kubectl exec -it ml-pod -- bash
  ```
  about: Runs a command (e.g., opens a bash shell) inside a pod (`ml-pod`). It’s like SSH-ing into a container for inspection.

- **`kubectl proxy`**
  ```bash
  kubectl proxy
  ```
  about: Starts a proxy to the Kubernetes API, often used to access the dashboard at `http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/`. Handy for monitoring.

---

## DeepOps and Ansible Commands

DeepOps uses Ansible to automate cluster setup (e.g., installing SLURM or Kubernetes). Here’s a typical command:

- **`ansible-playbook -i inventory playbook.yml`**
  ```bash
  ansible-playbook -i inventory.yml deploy-kubernetes.yml
  ```
  about: Runs an Ansible playbook (`deploy-kubernetes.yml`) to configure hosts listed in `inventory.yml`. In DeepOps, this might set up Kubernetes or SLURM across nodes. It’s the backbone of automated deployment.
