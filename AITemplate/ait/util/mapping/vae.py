import torch
from ...util import torch_dtype_from_str

def map_vae(pt_module, device="cuda", dtype="float16"):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_ait = {}
    for key, arr in pt_params.items():
        if key.startswith("encoder"):
            continue
        if key.startswith("quant"):
            continue
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            params_ait[key] = torch.permute(arr, [0, 2, 3, 1]).contiguous()
        elif key.endswith("proj_attn_weight"):
            prefix = key[: -len("proj_attn_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("to_out_0_weight"):
            prefix = key[: -len("to_out_0_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("proj_attn_bias"):
            prefix = key[: -len("proj_attn_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("to_out_0_bias"):
            prefix = key[: -len("to_out_0_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("query_weight"):
            prefix = key[: -len("query_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("to_q_weight"):
            prefix = key[: -len("to_q_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("query_bias"):
            prefix = key[: -len("query_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("to_q_bias"):
            prefix = key[: -len("to_q_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("key_weight"):
            prefix = key[: -len("key_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("key_bias"):
            prefix = key[: -len("key_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("value_weight"):
            prefix = key[: -len("value_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("value_bias"):
            prefix = key[: -len("value_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        elif key.endswith("to_k_weight"):
            prefix = key[: -len("to_k_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("to_v_weight"):
            prefix = key[: -len("to_v_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("to_k_bias"):
            prefix = key[: -len("to_k_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("to_v_bias"):
            prefix = key[: -len("to_v_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        else:
            params_ait[key] = arr

    return params_ait
