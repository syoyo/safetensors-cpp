import numpy as np
from safetensors.numpy import save_file

tensors = {
    "weight1": np.random.rand(8, 8),
    "weight2": np.random.rand(16, 16)
}

save_file(tensors, "model.safetensors")
