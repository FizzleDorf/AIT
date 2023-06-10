import torch
from ...util import torch_dtype_from_str


def map_controlnet(pt_mod, dim=320, device="cuda", dtype="float16"):
    if not isinstance(pt_mod, dict):
        pt_params = dict(pt_mod.named_parameters())
    else:
        pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
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
    params_ait["controlnet_cond_embedding_conv_in_weight"] = torch.nn.functional.pad(
        params_ait["controlnet_cond_embedding_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
    )
    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait
