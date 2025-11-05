import torch
import sys

def dynamic_quantize(model):
    
    if sys.platform == "darwin":
        print("Skipping dynamic quantization on macOS to avoid deepcopy issues.")
        return model
    else:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
   