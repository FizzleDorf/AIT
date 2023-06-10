import torch


class AITemplateModelWrapper(torch.nn.Module):
    def __init__(self, unet_ait_exe, alphas_cumprod, conditioning_key=None):
        super().__init__()
        self.unet_ait_exe = unet_ait_exe
        self.alphas_cumprod = alphas_cumprod
        #TODO: use the conditioning key
        self.conditioning_key = conditioning_key

    def apply_model(self, x, t, cond):
        timesteps_pt = t
        latent_model_input = x
        encoder_hidden_states = None
        down_block_residuals = None
        mid_block_residual = None
        #TODO: verify this is correct/match DiffusionWrapper (ddpm.py)
        if 'c_crossattn' in cond:
            encoder_hidden_states = cond['c_crossattn']
            encoder_hidden_states = encoder_hidden_states[0]
        if 'c_concat' in cond:
            encoder_hidden_states = cond['c_concat']
        if "control" in cond:
            down_block_residuals = cond["control"]["output"]
            mid_block_residual = cond["control"]["middle"][0]
        if encoder_hidden_states is None:
            raise f"conditioning missing, it should be one of these {cond.keys()}"
        return unet_inference(
            self.unet_ait_exe,
            latent_model_input=latent_model_input,
            timesteps=timesteps_pt,
            encoder_hidden_states=encoder_hidden_states,
            down_block_residuals=down_block_residuals,
            mid_block_residual=mid_block_residual,
        )


def unet_inference(
    exe_module, latent_model_input, timesteps, encoder_hidden_states, class_labels=None, down_block_residuals=None, mid_block_residual=None
):
    batch = latent_model_input.shape[0]
    height, width = latent_model_input.shape[2], latent_model_input.shape[3]
    timesteps_pt = timesteps
    inputs = {
        "input0": latent_model_input.permute((0, 2, 3, 1))
        .contiguous()
        .cuda()
        .half(),
        "input1": timesteps_pt.cuda().half(),
        "input2": encoder_hidden_states.cuda().half(),
    }
    if class_labels is not None:
        inputs["input3"] = class_labels.contiguous().cuda()
    if down_block_residuals is not None and mid_block_residual is not None:
        for i, y in enumerate(down_block_residuals):
            inputs[f"down_block_residual_{i}"] = y.permute((0, 2, 3, 1)).contiguous().cuda().half()
        inputs["mid_block_residual"] = mid_block_residual.permute((0, 2, 3, 1)).contiguous().cuda().half()
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch * 2
        shape[1] = height
        shape[2] = width
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    noise_pred = ys[0].permute((0, 3, 1, 2)).float()
    return noise_pred


def vae_inference(exe_module, vae_input, factor = 8):
    batch = vae_input.shape[0]
    height, width = vae_input.shape[2], vae_input.shape[3]
    inputs = [torch.permute(vae_input, (0, 2, 3, 1)).contiguous().cuda()]
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        shape[1] = height * factor
        shape[2] = width * factor
        ys.append(torch.empty(shape).cuda())
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    vae_out = ys[0].permute((0, 3, 1, 2)).float()
    return vae_out


def clip_inference(exe_module, input_ids, seqlen=77):
    batch = input_ids.shape[0]
    position_ids = torch.arange(seqlen).expand((batch, -1)).cuda()
    inputs = {
        "input0": input_ids,
        "input1": position_ids,
    }
    ys = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        ys.append(torch.empty(shape).cuda().half())
    exe_module.run_with_tensors(inputs, ys, graph_mode=False)
    return ys[0].float()
