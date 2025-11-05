# NeuralNetwork-Toolkit

A Python toolkit for dynamic quantization and pruning of neural networks to optimize memory and speed. Integrated with PyTorch for training and TensorRT for GPU inference.

## Features
- Dynamic quantization and pruning of neural networks
- Supports multiple model architectures
- TensorRT GPU inference for accelerated performance
- Benchmarking utilities for latency and throughput

 ##   Project Structure
 
data/ – sample input and output data
examples/ – (compress_and_run.py)

src/ – main source code
  - compression/ – quantization and pruning
  - inference/ – TensorRT and model inference
  - utils/ – benchmarking

tests/ – test scripts
requirements.txt – dependencies
README.md – project documentation

## Installation
```bash
# Clone the repo
git clone https://github.com/yourusername/adaptive-nn-compression.git
cd adaptive-nn-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt



## Note
- TensorRT is **only supported on Linux/Windows**.
