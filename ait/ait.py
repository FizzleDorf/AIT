from typing import Literal, Union

import torch
from safetensors.torch import load_file

from ait import AITLoader, clip_inference, unet_inference, vae_inference


class AIT:
    def __init__(self) -> None:
        self.modules = {}
        self.loader = AITLoader()
        self.supported = ['clip', 'controlnet', 'unet', 'vae']

    def load(self,
        aitemplate_path: str,
        hf_hub_or_path: str,
        module_type: str,
    ):
        if module_type == "clip":
            self.modules["clip"] = self.loader.load(aitemplate_path)
            clip = self.loader.diffusers_clip(hf_hub_or_path)
            self.modules["clip"] = self.loader.apply_clip(self.modules["clip"], clip)
        elif module_type == "controlnet":
            self.modules["controlnet"] = self.loader.load(aitemplate_path)
            controlnet = self.loader.diffusers_controlnet(hf_hub_or_path)
            self.modules["controlnet"] = self.loader.apply_controlnet(self.modules["controlnet"], controlnet)
        elif module_type == "unet":
            self.modules["unet"] = self.loader.load(aitemplate_path)
            unet = self.loader.diffusers_unet(hf_hub_or_path)
            self.modules["unet"] = self.loader.apply_unet(self.modules["unet"], unet)
        elif module_type == "vae":
            self.modules["vae"] = self.loader.load(aitemplate_path)
            vae = self.loader.diffusers_vae(hf_hub_or_path)
            self.modules["vae"] = self.loader.apply_vae(self.modules["vae"], vae)
        else:
            raise ValueError(f"module_type must be one of {self.supported}")

    def load_compvis(self,
        aitemplate_path: str,
        ckpt_path: str,
        module_type: str,
    ):
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        elif ckpt_path.endswith(".ckpt"):
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            raise ValueError("ckpt_path must be a .safetensors or .ckpt file")
        while "state_dict" in state_dict.keys():
            """
            yo dawg i heard you like state dicts so i put a state dict in your state dict

            apparently this happens in some models
            """
            state_dict = state_dict["state_dict"]
        if module_type == "clip":
            self.modules["clip"] = self.loader.load(aitemplate_path)
            clip = self.loader.compvis_clip(state_dict)
            self.modules["clip"] = self.loader.apply_clip(self.modules["clip"], clip)
        elif module_type == "controlnet":
            self.modules["controlnet"] = self.loader.load(aitemplate_path)
            controlnet = self.loader.compvis_controlnet(state_dict)
            self.modules["controlnet"] = self.loader.apply_controlnet(self.modules["controlnet"], controlnet)
        elif module_type == "unet":
            self.modules["unet"] = self.loader.load(aitemplate_path)
            unet = self.loader.compvis_unet(state_dict)
            self.modules["unet"] = self.loader.apply_unet(self.modules["unet"], unet)
        elif module_type == "vae":
            self.modules["vae"] = self.loader.load(aitemplate_path)
            vae = self.loader.compvis_vae(state_dict)
            self.modules["vae"] = self.loader.apply_vae(self.modules["vae"], vae)
        else:
            raise ValueError(f"module_type must be one of {self.supported}")


    def test_unet(self):
        if "unet" not in self.modules:
            raise ValueError("unet module not loaded")
        latent_model_input_pt = torch.randn(2, 4, 64, 64).cuda().half()
        text_embeddings_pt = torch.randn(2, 77, 768).cuda().half()
        timesteps_pt = torch.Tensor([1, 1]).cuda().half()
        output = unet_inference(
            self.modules["unet"],
            latent_model_input=latent_model_input_pt,
            timesteps=timesteps_pt,
            encoder_hidden_states=text_embeddings_pt
        )
        print(output.shape)
