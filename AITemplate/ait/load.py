from typing import Union

import json
import os
import lzma
import requests
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
      modules_path: str = "./modules/",
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
        self.modules_path = modules_path
        self.extension = "dll" if os.name == "nt" else "so"
        try:
            self.modules = json.load(open(f"{modules_path}/modules.json", "r"))
        except FileNotFoundError:
            raise FileNotFoundError(f"modules.json not found in {modules_path}")
        except json.decoder.JSONDecodeError:
            raise ValueError(f"modules.json in {modules_path} is not a valid json file")

    def download_module(self, sha256: str, url: str):
        module_path = f"{self.modules_path}/{sha256}.{self.extension}"
        temp_path = f"{self.modules_path}/{sha256}.{self.extension}.xz"
        if os.path.exists(module_path):
            return
        r = requests.get(url, stream=True)
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with lzma.open(temp_path, "rb") as f:
            with open(module_path, "wb") as g:
                g.write(f.read())
        os.remove(temp_path)

        

    def load_module(
        self, sha256: str, url: str
    ):
        module_path = f"{self.modules_path}/{sha256}.{self.extension}"
        download = False
        if not os.path.exists(module_path):
            download = True
        if download:
            self.download_module(sha256, url)
        return self.load(module_path)


    def filter_modules(self, operating_system: str, sd: str, cuda: str, batch_size: int, resolution: int, model_type: str, largest: bool = False):
        modules = [x for x in self.modules if x["os"] == operating_system and x["sd"] == sd and x["cuda"] == cuda and x["batch_size"] == batch_size and x["resolution"] >= resolution and model_type == x["model"]]
        if len(modules) == 0:
            raise ValueError(f"No modules found for {operating_system} {sd} {cuda} {batch_size} {resolution} {model_type}")
        print(f"Found {len(modules)} modules for {operating_system} {sd} {cuda} {batch_size} {resolution} {model_type}")
        modules = sorted(modules, key=lambda k: k['resolution'], reverse=largest)
        print(f"Using {modules[0]['sha256']}")
        return modules


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
            subfolder="unet" if not hf_hub_or_path.endswith("unet") else None,
            variant="fp16",
            use_safetensors=True,
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
