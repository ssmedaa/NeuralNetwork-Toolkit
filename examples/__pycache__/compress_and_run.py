import torch
from src.compression.pruning import prune_model
from src.compression.quantization import dynamic_quantize
from src.utils.benchmark import benchmark_model
from src.inference.tensorrt_inference import run_tensorrt_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Sample input
input_data = torch.randn(1, 10)
model = prune_model(model, amount=0.3)
model = dynamic_quantize(model)

print("Running a single forward pass...")
with torch.no_grad():
    output = model(input_data)

print("Forward pass output:", output)
benchmark_model(model, input_data)


output = run_tensorrt_model(model, input_data)
print("Inference output:", output)
