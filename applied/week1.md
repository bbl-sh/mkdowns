### Week 1 (operating Systems, Virtualization, and cloud)

Note - Most of the material is from the course material shared by the teachers of NPTEL

So, Let's begin by the **operating system (OS)** whcih is the big boss of the computer. It’s this software that sits in memory and manages everything—hardware, software, all the resources like CPU, memory, and I/O devices. It’s super privileged, meaning it gets to do cool stuff like write to the disk or talk over the network, stuff regular programs can’t touch. Programs need the OS to talk to hardware ‘cause they don’t know how to do it themselves. When a bunch of programs are running, the OS makes sure they share the hardware nicely—multiplexing and protection so they don’t mess each other up. Oh, and it’s got layers: user space, system call interface, and the OS itself sitting on the hardware.

Then there’s **virtualization**, which i have read about extensively in my last semesters(cloud computing) which is this neat trick where you make one real system look like a bunch of virtual ones—or sometimes the other way around. Like, one physical machine can act like multiple virtual machines (VMs), each with its own OS and apps—one-to-many style. Or, many physical machines can look like one big virtual thing—many-to-one. There’s even many-to-many! Uses? Running multiple OSes (even old ones), isolating apps for security, testing stuff, or moving servers around live. You’ve got two types: **process VMs** that virtualize the ABI (application binary interface) and run in user space, and **system VMs** that virtualize the ISA (instruction set architecture) with a hypervisor. Hypervisors are like OSes for OSes—Type 1 runs straight on hardware (bare metal, e.g., VMware ESX), Type 2 runs on top of a host OS (e.g., VMware Workstation). There’s also **para-virtualization**, where you tweak the guest OS to use hypercalls instead of trapping sensitive instructions—faster but needs OS mods.

---

### ai Accelerators and GPUs

**AI accelerators** are these fancy pieces of hardware built to speed up AI stuff—like neural networks and machine learning. There’s three main types:
- **CPUs**: General-purpose, good at lots of things, but not super fast for AI’s parallel tasks.
- **GPUs**: Awesome at parallel processing, perfect for breaking AI jobs into tons of little tasks.
- **FPGAs/ASICs**: Super specialized—FPGAs you can reprogram, ASICs are custom for one job.

They’re used in two spots: **data centers** for big, heavy computations, and **edge devices** (like phones or IoT) for quick, real-time stuff. How do they work? You load data to the CPU, ship it to the accelerator, process it in parallel, then send results back. GPUs shine here ‘cause they handle parallelism way better than CPUs, which do things one-by-one (serial). Like, for a reduction (e.g., summing numbers), CPU goes step-by-step, but GPU does it all at once—bam, faster for big data.

There’s cool hardware like the **NVIDIA DGX-1**, packed with multiple GPUs and Tensor Cores for deep learning. Tensor Cores do crazy fast matrix multiplications (e.g., 4x4 FP16 matrices in one clock cycle), which is clutch for AI. Benefits? Sustainability (less power waste), speed, scalability, mixing different hardware (heterogeneous architecture), and just being way more efficient than plain CPUs.

---

### Ai systems hardware

**Artificial Intelligence (AI)** is, per John McCarthy, “the science and engineering of making intelligent machines.” Deep learning’s a subset of machine learning where feature extraction gets automated—less human fiddling than classic ML. AI’s getting better thanks to three things:
- Algorithmic innovation (smarter math).
- Data (tons of it, supervised or interactive).
- Compute power (more juice for training).

AI algorithms, especially deep learning, are **compute-hungry**. Traditional systems have computation, communication, and storage/memory parts. The **memory hierarchy** goes from tiny, fast registers to big, slow storage—data access is a bottleneck, and moving data (especially off-chip to on-chip) eats tons of energy. Like, way more than the actual computing!

Back in the day, computation speed was skyrocketing, but around 2005, it hit a wall (Herb Sutter called it “the free lunch is over”). So, we shifted to concurrency and parallel processing—enter **specialized computation engines** like GPUs. NVIDIA’s **V100** has 640 Tensor Cores, cranking out 125 TOPS (tera operations per second) with mixed precision—super efficient for AI. There’s also **FPGAs**, programmable for specific tasks, and giants like **Cerebras WSE-2** with 2.6 trillion transistors for massive parallel work.

Market-wise, NVIDIA owns GPUs (72.8% in 2017), while Xilinx (53%) and Altera (36%) lead FPGAs. These specialized engines tackle AI’s data and energy issues way better than old-school setups.
