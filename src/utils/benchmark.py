import time
import torch

def benchmark_model(model, input_data, runs=100):

    model.eval()
    try:
    
        with torch.no_grad():
            _ = model(input_data)
    except Exception as e:
        print("Error during a test forward pass:", e)
        return

    start = time.time()
    for i in range(runs):
        with torch.no_grad():
            i = model(input_data)
    end = time.time()
    avg_time = (end - start) / runs
    
    print(f"Average inference time: {avg_time:.6f} seconds")
