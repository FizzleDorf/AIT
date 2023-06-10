import torch
from ait.util import torch_dtype_from_str

def map_vae(ait_module, pt_module, device="cuda", dtype="float16"):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_ait = {}
    for name, _ in ait_module.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            if (
                "conv" in name
                and "norm" not in name
                and name.endswith(".weight")
                and len(pt_params[name].shape) == 4
            ):
                params_ait[ait_name] = torch.permute(
                    pt_params[name], [0, 2, 3, 1]
                ).contiguous()
            else:
                params_ait[ait_name] = pt_params[name]
        elif name.endswith("attention.proj.weight"):
            prefix = name[: -len("attention.proj.weight")]
            pt_name = prefix + "proj_attn.weight"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_out.0.weight"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj.bias"):
            prefix = name[: -len("attention.proj.bias")]
            pt_name = prefix + "proj_attn.bias"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_out.0.bias"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.cu_length"):
            ...
        elif name.endswith("attention.proj_q.weight"):
            prefix = name[: -len("attention.proj_q.weight")]
            pt_name = prefix + "query.weight"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_q.weight"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_q.bias"):
            prefix = name[: -len("attention.proj_q.bias")]
            pt_name = prefix + "query.bias"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_q.bias"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_k.weight"):
            prefix = name[: -len("attention.proj_k.weight")]
            pt_name = prefix + "key.weight"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_k.weight"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_k.bias"):
            prefix = name[: -len("attention.proj_k.bias")]
            pt_name = prefix + "key.bias"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_k.bias"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_v.weight"):
            prefix = name[: -len("attention.proj_v.weight")]
            pt_name = prefix + "value.weight"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_v.weight"
                params_ait[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj_v.bias"):
            prefix = name[: -len("attention.proj_v.bias")]
            pt_name = prefix + "value.bias"
            if pt_name in pt_params:
                params_ait[ait_name] = pt_params[pt_name]
            else:
                pt_name = prefix + "to_v.bias"
                params_ait[ait_name] = pt_params[pt_name]
        else:
            pt_param = pt_module.get_parameter(name)
            params_ait[ait_name] = pt_param

    for key, arr in params_ait.items():
        params_ait[key] = arr.to(device, dtype=torch_dtype_from_str(dtype))

    return params_ait
