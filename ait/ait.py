from ait import AITLoader
from ait import unet_inference, clip_inference, vae_inference
from typing import Union, Literal
import torch

class AIT:
    def __init__(self) -> None:
        self.modules = {}
        self.loader = AITLoader()
        self.supported = ['clip', 'controlnet', 'unet', 'vae']

    def load(self,
        aitemplate_path,
        hf_hub_or_path,
        module_type,
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
