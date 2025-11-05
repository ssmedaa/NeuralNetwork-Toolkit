# NeuralNetwork-Toolkit

A Python toolkit for dynamic quantization and pruning of neural networks to optimize memory and speed. Integrated with PyTorch for training and TensorRT for GPU inference.

## Features
- Dynamic quantization and pruning of neural networks
- Supports multiple model architectures
- TensorRT GPU inference for accelerated performance
- Benchmarking utilities for latency and throughput

##   Project Structure
NeuralNetwork-Toolkit/
├── data/                         
│   ├── input/sample_input.npy
│   ├── output/sample_output.npy
│   ├── sample_data.py
├── examples/
│   └── compress_and_run.py     
├── src/
│   ├── compression/
│   │   ├── quantization.py
│   │   ├── pruning.py
│   ├── inference/
│   │  └──  tensorrt_inference.py
│   └── utils/
│       └── benchmark.py
├── tests/       
│       ├── test_pruning.py
│       ├── test_quantization.py
├── requirements.txt
└── README.md

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



## ** Note **


## Notes
- TensorRT is **only supported on Linux/Windows**.
