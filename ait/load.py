from typing import Union

import torch
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel

from ait.module import Model
from ait.util import torch_dtype_from_str
from ait.util.mapping import map_clip, map_controlnet, map_unet, map_vae


class AITLoader:
    def __init__(self,
      num_runtimes: int = 1,
      device: Union[str, torch.device] = "cuda",
      dtype: str = "float16",
    ) -> None:
        """
        device and dtype can be overriden at the function level
        device must be a cuda device
        """
        self.device = device
        self.dtype = dtype
        self.num_runtimes = num_runtimes

    def load(
        self,
        path: str,
    ) -> Model:
        aitemplate = Model(lib_path=path, num_runtimes=self.num_runtimes)
        return aitemplate

    def diffusers_unet(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "unet",
        revision: str = "fp16",
    ) -> UNet2DConditionModel:
        unet = UNet2DConditionModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
        return unet
    
    def diffusers_vae(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "vae",
        revision: str = "fp16",
    ) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
        return vae

    def diffusers_controlnet(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = None,
        revision: str = None,
    ) -> ControlNetModel:
        controlnet = ControlNetModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
        return controlnet
    
    def diffusers_clip(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "text_encoder",
        revision: str = "fp16",
    ) -> CLIPTextModel:
        clip = CLIPTextModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
        return clip

    def apply(
        self,
        aitemplate_module: Model,
        ait_params: dict,
    ) -> Model:
        aitemplate_module.set_many_constants_with_tensors(ait_params)
        aitemplate_module.fold_constants()
        return aitemplate_module

    def apply_unet(
        self,
        aitemplate_module: Model,
        unet: Union[UNet2DConditionModel, dict],
        in_channels: int = None,
        conv_in_key: str = None,
        dim: int = 320,
        device: Union[str, torch.device] = None,
        dtype: str = None,
    ) -> Model:
        """
        you don't need to set in_channels or conv_in_key unless
        you are experimenting with other UNets
        """
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_params = map_unet(unet, in_channels=in_channels, conv_in_key=conv_in_key, dim=dim, device=device, dtype=dtype)
        return self.apply(aitemplate_module, ait_params)

    def apply_clip(
        self,
        aitemplate_module: Model,
        clip: Union[CLIPTextModel, dict],
        device: Union[str, torch.device] = None,
        dtype: str = None,
    ) -> Model:
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_params = map_clip(clip, device=device, dtype=dtype)
        return self.apply(aitemplate_module, ait_params)

    def apply_controlnet(
        self,
        aitemplate_module: Model,
        controlnet: Union[ControlNetModel, dict],
        dim: int = 320,
        device: Union[str, torch.device] = None,
        dtype: str = None,
    ) -> Model:
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_params = map_controlnet(controlnet, dim=dim, device=device, dtype=dtype)
        return self.apply(aitemplate_module, ait_params)

    """
block_out_channels onwards are required to
instantiate the AITemplate AutoencoderKL
for parameter mapping
a better mapping method needs to be investigated

the values should match the architecture
of the compiled module
should not be an issue unless you are 
experimenting with other vae architectures
i.e. x4-upscaler
values can be found from diffusers config
or accessed from PT AutoencoderKL.config
    """
    def apply_vae(
        self,
        aitemplate_module: Model,
        vae: Union[AutoencoderKL, dict],
        device: Union[str, torch.device] = None,
        dtype: str = None,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        sample_size=512,
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        input_size=(64, 64),
    ) -> Model:
        try:
            import aitemplate
        except ImportError:
            raise ImportError("aitemplate is required to apply vae to ait")
        from ait.modeling import AIT_AutoencoderKL
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_vae = AIT_AutoencoderKL(
            1,
            input_size[0],
            input_size[1],
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            dtype=dtype
        )
        ait_vae.name_parameter_tensor()
        ait_params = map_vae(ait_vae, vae, device=device, dtype=dtype)
        return self.apply(aitemplate_module, ait_params)
