from typing import Literal, Union

import torch
from safetensors.torch import load_file

from .load import AITLoader
from .module import Model
from .inference import clip_inference, unet_inference, vae_inference, controlnet_inference


class AIT:
    def __init__(self, path: str) -> None:
        self.modules = {}
        self.unet = {}
        self.vae = {}
        self.controlnet = {}
        self.clip = {}
        self.control_net = None
        self.loader = AITLoader(path)
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
        elif module_type == "vae_encode":
            self.modules["vae_encode"] = self.loader.load(aitemplate_path)
            vae = self.loader.diffusers_vae(hf_hub_or_path)
            self.modules["vae_encode"] = self.loader.apply_vae(self.modules["vae_encode"], vae, encoder=True)
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
        elif module_type == "vae_encode":
            self.modules["vae_encode"] = self.loader.load(aitemplate_path)
            vae = self.loader.compvis_vae(state_dict)
            self.modules["vae_encode"] = self.loader.apply_vae(self.modules["vae_encode"], vae, encoder=True)
        else:
            raise ValueError(f"module_type must be one of {self.supported}")


    def test_unet(
        self,
        batch_size: int = 2,
        latent_channels: int = 4,
        height: int = 64,
        width: int = 64,
        hidden_dim: int = 768,
        sequence_length: int = 77,
        dtype="float16",
        device="cuda",
        benchmark: bool = False,
    ):
        if "unet" not in self.modules:
            raise ValueError("unet module not loaded")
        latent_model_input_pt = torch.randn(batch_size, latent_channels, height, width).to(device)
        text_embeddings_pt = torch.randn(batch_size, sequence_length, hidden_dim).to(device)
        timesteps_pt = torch.Tensor([1] * batch_size).to(device)
        if dtype == "float16":
            latent_model_input_pt = latent_model_input_pt.half()
            text_embeddings_pt = text_embeddings_pt.half()
            timesteps_pt = timesteps_pt.half()
        output = unet_inference(
            self.modules["unet"],
            latent_model_input=latent_model_input_pt,
            timesteps=timesteps_pt,
            encoder_hidden_states=text_embeddings_pt,
            benchmark=benchmark,
        )
        print(output.shape)
        return output

    def test_vae_encode(
        self,
        batch_size: int = 1,
        channels: int = 3,
        height: int = 512,
        width: int = 512,
        dtype="float16",
        device="cuda",
    ):
        if "vae_encode" not in self.modules:
            raise ValueError("vae module not loaded")
        vae_input = torch.randn(batch_size, channels, height, width).to(device)
        if dtype == "float16":
            vae_input = vae_input.half()
        output = vae_inference(
            self.modules["vae_encode"],
            vae_input=vae_input,
            encoder=True,
        )
        print(output.shape)
        return output


    def test_vae(
        self,
        batch_size: int = 1,
        latent_channels: int = 4,
        height: int = 64,
        width: int = 64,
        dtype="float16",
        device="cuda",
    ):
        if "vae" not in self.modules:
            raise ValueError("vae module not loaded")
        vae_input = torch.randn(batch_size, latent_channels, height, width).to(device)
        if dtype == "float16":
            vae_input = vae_input.half()
        output = vae_inference(
            self.modules["vae"],
            vae_input=vae_input,
        )
        print(output.shape)
        return output
    
    def test_clip(
        self,
        batch_size: int = 1,
        sequence_length: int = 77,
        tokenizer=None,
    ):
        if "clip" not in self.modules:
            raise ValueError("clip module not loaded")
        try:
            from transformers import CLIPTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers with `pip install transformers` to use this script."
            )
        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_input = tokenizer(
            ["a photo of an astronaut riding a horse on mars"] * batch_size,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input["input_ids"].cuda()
        output = clip_inference(
            self.modules["clip"],
            input_ids=input_ids,
            seqlen=sequence_length,
        )
        print(output.shape)
        return output

    def test_controlnet(
        self,
        batch_size: int = 2,
        latent_channels: int = 4,
        latent_height: int = 64,
        latent_width: int = 64,
        hidden_dim: int = 768,
        sequence_length: int = 77,
        control_height: int = 512,
        control_width: int = 512,
        control_channels: int = 3,
        device="cuda",
        dtype="float16",
    ):
        latent_model_input_pt = torch.randn(batch_size, latent_channels, latent_height, latent_width).to(device)
        text_embeddings_pt = torch.randn(batch_size, sequence_length, hidden_dim).to(device)
        timesteps_pt = torch.Tensor([1] * batch_size).to(device)
        controlnet_input_pt = torch.randn(batch_size, control_channels, control_height, control_width).to(device)
        if dtype == "float16":
            latent_model_input_pt = latent_model_input_pt.half()
            text_embeddings_pt = text_embeddings_pt.half()
            timesteps_pt = timesteps_pt.half()
            controlnet_input_pt = controlnet_input_pt.half()
        down_block_residuals, mid_block_residual = controlnet_inference(
            self.modules["controlnet"],
            latent_model_input=latent_model_input_pt,
            timesteps=timesteps_pt,
            encoder_hidden_states=text_embeddings_pt,
            controlnet_cond=controlnet_input_pt,
        )
        for down_block_residual in down_block_residuals:
            print(down_block_residual.shape)
        print(mid_block_residual.shape)
        return down_block_residuals, mid_block_residual
