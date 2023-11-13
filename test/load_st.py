import safetensors
import torch

with safetensors.safe_open("../build/example.safetensors", framework='pt', device='cpu') as f:

    for key in f.keys():
        print(f.get_tensor(key))
