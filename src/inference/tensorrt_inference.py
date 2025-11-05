import torch
import sys

def run_tensorrt_model(model, input_data):
    
    if sys.platform == "darwin":
        print("TensorRT not supported on macOS. Running PyTorch inference instead.")
        model.eval()
        with torch.no_grad():
            return model(input_data)
    else:
        try:
            import tensorrt as trt
            # Add TensorRT inference code for Linux/Windows here
            print("TensorRT inference not implemented yet.")
            model.eval()
            with torch.no_grad():
                return model(input_data)
        except ImportError:
            print("TensorRT not installed. Running PyTorch inference instead.")
            model.eval()
            with torch.no_grad():
                return model(input_data)
