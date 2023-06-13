import os
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.sd
import comfy.k_diffusion.external as k_diffusion_external
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import torch
import contextlib
import sys
import time

from .ait.inference import AITemplateModelWrapper
from .ait import AIT
from .ait.inference import clip_inference, unet_inference, vae_inference, controlnet_inference

supported_ait_extensions = set(['.so', '.xz'])
base_path = os.path.dirname(os.path.realpath(__file__))
modules_dir = os.path.join(base_path, "modules")
folder_names_and_paths = {}
folder_names_and_paths["aitemplate"] = ([modules_dir], supported_ait_extensions)
filename_list_cache = {}
current_loaded_model = None
vram_state = None

AITemplate = AIT()

def get_full_path(folder_name, filename):
    global folder_names_and_paths
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path

    return None

def recursive_search(directory):
    if not os.path.isdir(directory):
        return [], {}
    result = []
    dirs = {directory: os.path.getmtime(directory)}
    for root, subdir, file in os.walk(directory, followlinks=True):
        for filepath in file:
            #we os.path,join directory with a blank string to generate a path separator at the end.
            result.append(os.path.join(root, filepath).replace(os.path.join(directory,''),''))
        for d in subdir:
            path = os.path.join(root, d)
            dirs[path] = os.path.getmtime(path)
    return result, dirs

def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions, files)))


def get_filename_list_(folder_name):
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x)
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}

    return (sorted(list(output_list)), output_folders, time.perf_counter())

def cached_filename_list_(folder_name):
    global filename_list_cache
    global folder_names_and_paths
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]
    if time.perf_counter() < (out[2] + 0.5):
        return out
    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:
                return None

    return out

def get_filename_list(folder_name):
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    return list(out[0])


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False):
    global current_loaded_model
    global vram_state
    global AITemplate
    use_aitemplate = isinstance(model, tuple)
    if use_aitemplate:
        model, keep_loaded, aitemplate_path = model
    device = comfy.model_management.get_torch_device()

    if use_aitemplate and keep_loaded == "disable" and "unet" not in AITemplate.modules:
        AITemplate.modules["unet"] = AITemplate.loader.load(aitemplate_path)

    if noise_mask is not None:
        noise_mask = comfy.sample.prepare_mask(noise_mask, noise.shape, device)

    if use_aitemplate:
        apply_aitemplate_weights = current_loaded_model != model or keep_loaded == "disable"
        vram_state = comfy.model_management.vram_state
        comfy.model_management.vram_state = comfy.model_management.VRAMState.DISABLED
    else:
        if vram_state is not None:
            comfy.model_management.vram_state = vram_state

    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    if use_aitemplate:
        current_loaded_model = model
        real_model.alphas_cumprod = real_model.alphas_cumprod.float()
        if apply_aitemplate_weights:
            AITemplate.modules["unet"] = AITemplate.loader.apply_unet(
                aitemplate_module=AITemplate.modules["unet"],
                unet=AITemplate.loader.compvis_unet(real_model.state_dict())
            )

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
    negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

    models = comfy.sample.load_additional_models(positive, negative)

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
    if use_aitemplate:
        model_wrapper = AITemplateModelWrapper(AITemplate.modules["unet"], real_model.alphas_cumprod)
        sampler.model_denoise = comfy.samplers.CFGNoisePredictor(model_wrapper)
        if real_model.parameterization == "v":
            sampler.model_wrap = comfy.samplers.CompVisVDenoiser(sampler.model_denoise, quantize=True)
        else:
            sampler.model_wrap = k_diffusion_external.CompVisDenoiser(sampler.model_denoise, quantize=True)
        sampler.model_wrap.parameterization = sampler.model.parameterization
        sampler.model_k = comfy.samplers.KSamplerX0Inpaint(sampler.model_wrap)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar)
    samples = samples.cpu()

    comfy.sample.cleanup_additional_models(models)

    if use_aitemplate and keep_loaded == "disable":
        AITemplate.modules.pop("unet")
        del sampler
        torch.cuda.empty_cache()

    return samples

comfy.sample.sample = sample



class ControlNet:
    def __init__(self, control_model, device=None):
        self.aitemplate = None
        self.control_model = control_model
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        if device is None:
            device = comfy.model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None

    def aitemplate_controlnet(
        self, latent_model_input, timesteps, encoder_hidden_states, controlnet_cond
    ):
        if self.aitemplate is None:
            raise RuntimeError("No aitemplate loaded")
        return controlnet_inference(
            exe_module=self.aitemplate,
            latent_model_input=latent_model_input,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

    def get_control(self, x_noisy, t, cond_txt, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond_txt, batched_number)

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = comfy.sd.broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.aitemplate is None:
            if self.control_model.dtype == torch.float16:
                precision_scope = torch.autocast
            else:
                precision_scope = contextlib.nullcontext

            with precision_scope(comfy.model_management.get_autocast_device(self.device)):
                self.control_model = comfy.model_management.load_if_low_vram(self.control_model)
                control = self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=t, context=cond_txt)
                self.control_model = comfy.model_management.unload_if_low_vram(self.control_model)
        else:
            control = self.aitemplate_controlnet(x_noisy, t, cond_txt, self.cond_hint)
        out = {'middle':[], 'output': []}
        autocast_enabled = torch.is_autocast_enabled()

        for i in range(len(control)):
            if i == (len(control) - 1):
                key = 'middle'
                index = 0
            else:
                key = 'output'
                index = i
            x = control[i]
            x *= self.strength
            if x.dtype != output_dtype and not autocast_enabled:
                x = x.to(output_dtype)

            if control_prev is not None and key in control_prev:
                prev = control_prev[key][index]
                if prev is not None:
                    x += prev
            out[key].append(x)
        if control_prev is not None and 'input' in control_prev:
            out['input'] = control_prev['input']
        return out

    def set_cond_hint(self, cond_hint, strength=1.0):
        self.cond_hint_original = cond_hint
        self.strength = strength
        return self

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None

    def copy(self):
        c = ControlNet(self.control_model)
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        return c

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        out.append(self.control_model)
        return out

comfy.sd.ControlNet = ControlNet

class AITemplateLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "aitemplate_module": (get_filename_list("aitemplate"), ),
                              "keep_loaded": (["enable", "disable"], ),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_aitemplate"

    CATEGORY = "loaders"

    def load_aitemplate(self, model, aitemplate_module, keep_loaded):
        global AITemplate
        aitemplate_path = get_full_path("aitemplate", aitemplate_module)
        AITemplate.modules["unet"] = AITemplate.loader.load(aitemplate_path)
        return ((model,keep_loaded,aitemplate_path),)



class AITemplateVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "pixels": ("IMAGE", ),
            "vae": ("VAE", ),
            "aitemplate_module": (get_filename_list("aitemplate"), ),
            "keep_loaded": (["enable", "disable"], ),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def encode(self, vae, pixels, aitemplate_module, keep_loaded):
        global AITemplate
        if "vae_encode" not in AITemplate.modules:
            aitemplate_path = get_full_path("aitemplate", aitemplate_module)
            AITemplate.modules["vae_encode"] = AITemplate.loader.load(aitemplate_path)
            AITemplate.modules["vae_encode"] = AITemplate.loader.apply_vae(
                aitemplate_module=AITemplate.modules["vae_encode"],
                vae=AITemplate.loader.compvis_vae(vae.first_stage_model.state_dict()),
                encoder=True,
            )
        pixels = self.vae_encode_crop_pixels(pixels)
        pixels = pixels[:,:,:,:3]
        pixels = pixels.movedim(-1, 1)
        pixels = 2. * pixels - 1.
        moments = vae_inference(AITemplate.modules["vae_encode"], pixels, encoder=True)
        posterior = DiagonalGaussianDistribution(moments)
        samples = posterior.sample() * vae.scale_factor
        if keep_loaded == "disable":
            AITemplate.modules.pop("vae_encode")
            torch.cuda.empty_cache()
        return ({"samples":samples}, )


class AITemplateVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    { 
                    "vae": ("VAE",),
                    "aitemplate_module": (get_filename_list("aitemplate"), ),
                    "keep_loaded": (["enable", "disable"], ),
                    "samples": ("LATENT", ), "vae": ("VAE", )
                    }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, aitemplate_module, keep_loaded, samples):
        global AITemplate
        if "vae" not in AITemplate.modules:
            aitemplate_path = get_full_path("aitemplate", aitemplate_module)
            AITemplate.modules["vae"] = AITemplate.loader.load(aitemplate_path)
            AITemplate.modules["vae"] = AITemplate.loader.apply_vae(
            aitemplate_module=AITemplate.modules["vae"],
                vae=AITemplate.loader.compvis_vae(vae.first_stage_model.state_dict()),
            )
        output = (torch.clamp((vae_inference(AITemplate.modules["vae"], 1. / vae.scale_factor * samples["samples"]) + 1.0) / 2.0, min=0.0, max=1.0).cpu().movedim(1,-1), )
        if keep_loaded == "disable":
            AITemplate.modules.pop("vae")
            torch.cuda.empty_cache()
        return output


class AITemplateControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net": ("CONTROL_NET",),
                              "aitemplate_module": (get_filename_list("aitemplate"), ),
                              }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_aitemplate_controlnet"

    CATEGORY = "loaders"

    def load_aitemplate_controlnet(self, control_net, aitemplate_module):
        aitemplate_path = get_full_path("aitemplate", aitemplate_module)
        aitemplate = AITemplate.loader.load(aitemplate_path)
        aitemplate = AITemplate.loader.apply_controlnet(
            aitemplate_module=aitemplate,
            controlnet=AITemplate.loader.compvis_controlnet(control_net.control_model.state_dict())
        )
        control_net.aitemplate = aitemplate
        return (control_net,)

