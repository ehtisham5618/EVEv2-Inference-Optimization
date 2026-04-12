# Profiling-Guided Inference Optimization for EVEv2

A profiling-guided inference optimization project for the EVEv2 7B Vision-Language Model implemented in Python using PyTorch. This project focuses on optimizing inference efficiency on low-resource hardware (such as NVIDIA Tesla T4) using various optimization techniques.

---

## 🚀 Features

- **Identified Bottlenecks**: Detected CPU→GPU transfer bottleneck (72% overhead) using PyTorch Profiler.
- **Mixed Precision**: Applied FP16 mixed precision optimization.
- **Model Sharding**: Implemented model sharding to distribute the load across GPUs.
- **Performance Gains**: Improved throughput by 1.185× and reduced overall execution time by 3.4 hours.
- **Accuracy Maintained**: Sustained a high F1 score of 83.78% on the POPE Hallucination Benchmark.
- **Efficient Data Handling**: Integrated batched inference and asynchronous data loading.

---

## 🛠 Technologies Used

- Python
- PyTorch & PyTorch Profiler
- HuggingFace Accelerate
- Jupyter Notebook

---

## 📁 How to Run (Windows)

- Clone the repository and navigate to the project directory.
- Ensure that Python 3.8+ and Jupyter Notebook are installed.
- Install the required dependencies (PyTorch, Accelerate, etc.):
```bash
pip install torch torchvision torchaudio accelerate transformers jupyter
```
- Open the Jupyter Notebook:
```bash
jupyter notebook Code_File.ipynb
```
- Run the cells to see the profiling results and the inference optimizations.

---

## 🧠 Concepts Practiced

This project was developed focusing on Parallel and Distributed Computing and Efficient AI Systems, applying several advanced concepts:

- Model Parallelism & Pipeline Parallelism
- SIMD GPU Execution
- Inference Optimization
- GPU Memory Bottleneck Identification
- Profiling-Guided Systems Design
- FP16 Mixed Precision

---

## 👨💻 Author

Ehtisham Abid
