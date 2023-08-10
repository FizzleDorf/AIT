import torch
from ...util import torch_dtype_from_str

def map_unet(pt_mod, in_channels=None, conv_in_key=None, dim=320, device="cuda", dtype="float16"):
    if in_channels is not None and conv_in_key is None:
        raise ValueError("conv_in_key must be specified if in_channels is not None for padding")
    if not isinstance(pt_mod, dict):
        pt_params = dict(pt_mod.named_parameters())
    else:
        pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "")
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    if conv_in_key is not None:
        if in_channels % 4 != 0:
            pad_by = 4 - (in_channels % 4)
            params_ait[conv_in_key] = torch.functional.F.pad(params_ait[conv_in_key], (0, pad_by))

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).to(device, dtype=torch_dtype_from_str(dtype))
    )

    return params_ait
