try:
    from transformers import CLIPTextConfig, CLIPTextModel
except ImportError:
    raise ImportError(
        "Please install transformers with `pip install transformers` to use this script."
    )

import torch
from ...util import torch_dtype_from_str


def map_clip(pt_mod, device="cuda", dtype="float16"):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr
    return params_ait
