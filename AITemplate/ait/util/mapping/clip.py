from transformers import CLIPTextConfig, CLIPTextModel

from ...util import torch_dtype_from_str


def map_clip(pt_mod, device="cuda", dtype="float16"):
    if isinstance(pt_mod, dict):
        """
        TODO: investigate whether this dependency can be removed
        possibly:
        * position_ids could be created in another way
        * check what is missing from state dict as received here vs .named_parameters()
            * create the missing tensors another way if possible 
        """
        if "text_model.encoder.layers.22.layer_norm1.weight" in pt_mod.keys():
            clip_text_config = CLIPTextConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=23,
            projection_dim=512,
            hidden_act="gelu"
        )
        else:
            clip_text_config = CLIPTextConfig(
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_hidden_layers=12,
                projection_dim=768,
            )
        clip_text_model = CLIPTextModel(clip_text_config)
        pt_mod["text_model.embeddings.position_ids"] = clip_text_model.text_model.embeddings.get_buffer("position_ids")
        clip_text_model.load_state_dict(pt_mod)
        pt_params = dict(clip_text_model.named_parameters())
    else:
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
