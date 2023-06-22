from typing import Union

import torch
try:
    from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
except ImportError:
    pass
try:
    from transformers import CLIPTextModel
except ImportError:
    pass

from .module import Model
from .util import torch_dtype_from_str, convert_ldm_unet_checkpoint, convert_text_enc_state_dict, convert_ldm_vae_checkpoint
from .util.mapping import map_clip, map_controlnet, map_unet, map_vae


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
        return Model(lib_path=path, num_runtimes=self.num_runtimes)

    def compvis_unet(
        self,
        state_dict: dict,
    ) -> dict:
        """
        removes:
        model.diffusion_model.
        diffusion_model.
        from keys if present before conversion
        """
        return convert_ldm_unet_checkpoint(state_dict)
    
    def compvis_clip(
        self,
        state_dict: dict,
    ) -> dict:
        """
        removes:
        cond_stage_model.transformer.
        cond_stage_model.model.
        from keys if present before conversion
        """
        return convert_text_enc_state_dict(state_dict)
    
    def compvis_vae(
        self,
        state_dict: dict,
    ) -> dict:
        """
        removes:
        first_stage_model.
        from keys if present before conversion
        """
        return convert_ldm_vae_checkpoint(state_dict)
    
    def compvis_controlnet(
        self,
        state_dict: dict,
    ) -> dict:
        """
        removes:
        control_model.
        from keys if present before conversion
        """
        return convert_ldm_unet_checkpoint(state_dict, controlnet=True)

    def diffusers_unet(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "unet",
        revision: str = "fp16",
    ):
        return UNet2DConditionModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
    
    def diffusers_vae(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "vae",
        revision: str = "fp16",
    ):
        return AutoencoderKL.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )

    def diffusers_controlnet(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = None,
        revision: str = None,
    ):
        return ControlNetModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )
    
    def diffusers_clip(
        self,
        hf_hub_or_path: str,
        dtype: str = "float16",
        subfolder: str = "text_encoder",
        revision: str = "fp16",
    ):
        return CLIPTextModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            revision=revision,
            torch_dtype=torch_dtype_from_str(dtype)
        )

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
        unet,#: Union[UNet2DConditionModel, dict],
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
        clip,#: Union[CLIPTextModel, dict],
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
        controlnet,#: Union[ControlNetModel, dict],
        dim: int = 320,
        device: Union[str, torch.device] = None,
        dtype: str = None,
    ) -> Model:
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_params = map_controlnet(controlnet, dim=dim, device=device, dtype=dtype)
        return self.apply(aitemplate_module, ait_params)

    def apply_vae(
        self,
        aitemplate_module: Model,
        vae,#: Union[AutoencoderKL, dict],
        device: Union[str, torch.device] = None,
        dtype: str = None,
        encoder: bool = False,
    ) -> Model:
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        ait_params = map_vae(vae, device=device, dtype=dtype, encoder=encoder)
        return self.apply(aitemplate_module, ait_params)
