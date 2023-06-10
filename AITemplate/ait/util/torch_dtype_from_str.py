import torch

def torch_dtype_from_str(dtype: str):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype not supported yet!")
    return torch_dtype
