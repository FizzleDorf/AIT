from typing import List

import torch

from .module import Model


class AITemplateModelWrapper(torch.nn.Module):
    def __init__(
        self,
        unet_ait_exe: Model,
        alphas_cumprod: torch.Tensor,
    ):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod
        self.unet_ait_exe = unet_ait_exe

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_crossattn = None,
        c_concat = None,
        control = None,
        c_adm = None,
        transformer_options = None,
    ):
        timesteps_pt = t
        latent_model_input = x
        encoder_hidden_states = None
        down_block_residuals = None
        mid_block_residual = None
        add_embeds = None
        if c_crossattn is not None:
            encoder_hidden_states = c_crossattn
        if c_concat is not None:
            latent_model_input = torch.cat([x] + c_concat, dim=1)
        if control is not None:
            down_block_residuals = control["output"]
            mid_block_residual = control["middle"][0]
        if c_adm is not None:
            add_embeds = c_adm
        return unet_inference(
            self.unet_ait_exe,
            latent_model_input=latent_model_input,
            timesteps=timesteps_pt,
            encoder_hidden_states=encoder_hidden_states,
            down_block_residuals=down_block_residuals,
            mid_block_residual=mid_block_residual,
            add_embeds=add_embeds,
        )


def unet_inference(
    exe_module: Model,
    latent_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    class_labels: torch.Tensor = None,
    down_block_residuals: List[torch.Tensor] = None,
    mid_block_residual: torch.Tensor = None,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
    add_embeds: torch.Tensor = None,
):
    batch = latent_model_input.shape[0]
    height, width = latent_model_input.shape[2], latent_model_input.shape[3]
    timesteps_pt = timesteps.expand(batch)
    inputs = {
        "latent_model_input": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .to(device),
        "timesteps": timesteps_pt.to(device),
        "encoder_hidden_states": encoder_hidden_states.to(device),
    }
    if class_labels is not None:
        inputs["class_labels"] = class_labels.contiguous().to(device)
    if down_block_residuals is not None and mid_block_residual is not None:
        for i, y in enumerate(down_block_residuals):
            inputs[f"down_block_residual_{i}"] = y.permute((0, 2, 3, 1)).contiguous().to(device)
        inputs["mid_block_residual"] = mid_block_residual.permute((0, 2, 3, 1)).contiguous().to(device)
    if add_embeds is not None:
        inputs["add_embeds"] = add_embeds.to(device)
    if dtype == "float16":
        for k, v in inputs.items():
            if k == "class_labels ":
                continue
            inputs[k] = v.half()
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        shape[1] = height
        shape[2] = width
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    noise_pred = ys[0].permute((0, 3, 1, 2)).float()
    if benchmark:
        t, _, _ = exe_module.benchmark_with_tensors(
            inputs=inputs,
            outputs=ys,
            count=50,
            repeat=4,
        )
        print(f"unet latency: {t} ms, it/s: {1000 / t}")
    return noise_pred.cpu()


def controlnet_inference(
    exe_module: Model,
    latent_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    controlnet_cond: torch.Tensor,
    add_embeds: torch.Tensor = None,
    device: str = "cuda",
    dtype: str = "float16",
    benchmark: bool = False,
):
    if controlnet_cond.shape[0] != latent_model_input.shape[0]:
        controlnet_cond = controlnet_cond.expand(latent_model_input.shape[0], -1, -1, -1)
    if type(encoder_hidden_states) == dict:
        encoder_hidden_states = encoder_hidden_states['c_crossattn']
    inputs = {
        "latent_model_input": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .to(device),
        "timesteps": timesteps.to(device),
        "encoder_hidden_states": encoder_hidden_states.to(device),
        "control_hint": controlnet_cond.permute((0, 2, 3, 1)).contiguous().to(device),
    }
    if add_embeds is not None:
        inputs["add_embeds"] = add_embeds.to(device)
    if dtype == "float16":
        for k, v in inputs.items():
            inputs[k] = v.half()
    ys = {}
    for name, idx in exe_module.get_output_name_to_index_map().items():
        shape = exe_module.get_output_maximum_shape(idx)
        shape = torch.empty(shape).to(device)
        if dtype == "float16":
            shape = shape.half()
        ys[name] = shape
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    ys = {k: y.permute((0, 3, 1, 2)).float() for k, y in ys.items()}
    if benchmark:
        ys = {}
        for name, idx in exe_module.get_output_name_to_index_map().items():
            shape = exe_module.get_output_maximum_shape(idx)
            shape = torch.empty(shape).to(device)
            if dtype == "float16":
                shape = shape.half()
            ys[name] = shape
        t, _, _ = exe_module.benchmark_with_tensors(
            inputs=inputs,
            outputs=ys,
            count=50,
            repeat=4,
        )
        print(f"controlnet latency: {t} ms, it/s: {1000 / t}")
    return ys



def vae_inference(
    exe_module: Model,
    vae_input: torch.Tensor,
    factor: int = 8,
    device: str = "cuda",
    dtype: str = "float16",
    encoder: bool = False,
    latent_channels: int = 4,
):
    batch = vae_input.shape[0]
    height, width = vae_input.shape[2], vae_input.shape[3]
    if encoder:
        height = height // factor
        width = width // factor
    else:
        height = height * factor
        width = width * factor
    input_name = "pixels" if encoder else "latent"
    inputs = {
        input_name: torch.permute(vae_input, (0, 2, 3, 1))
        .contiguous()
        .to(device),
    }
    if encoder:
        sample = torch.randn(batch, latent_channels, height, width)
        inputs["random_sample"] = torch.permute(sample, (0, 2, 3, 1)).contiguous().to(device)
    if dtype == "float16":
        for k, v in inputs.items():
            inputs[k] = v.half()
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        shape[1] = height
        shape[2] = width
        ys.append(torch.empty(shape).to(device))
        if dtype == "float16":
            ys[i] = ys[i].half()
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    vae_out = ys[0].permute((0, 3, 1, 2)).cpu().float()
    return vae_out


def clip_inference(
    exe_module: Model,
    input_ids: torch.Tensor,
    seqlen: int = 77,
    device: str = "cuda",
    dtype: str = "float16",
):
    batch = input_ids.shape[0]
    input_ids = input_ids.to(device)
    position_ids = torch.arange(seqlen).expand((batch, -1)).to(device)
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        ys.append(torch.empty(shape).to(device))
        if dtype == "float16":
            ys[i] = ys[i].half()
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    return ys[0].cpu().float()
