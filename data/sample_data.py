import numpy as np
import os

os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)


input_data = np.random.randn(5, 10).astype(np.float32)
np.save("data/input/sample_input.npy", input_data)


output_data = np.random.randn(5, 1).astype(np.float32)
np.save("data/output/sample_output.npy", output_data)

print("Sample data created")

