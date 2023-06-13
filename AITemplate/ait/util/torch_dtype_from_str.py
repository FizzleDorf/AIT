import torch

def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)