# AIT
 
Experimental usage of [AITemplate](https://github.com/facebookincubator/AITemplate).

Also functions as a ComfyUI custom node
* Drop `AITemplate` folder in `ComfyUI/custom_nodes/`
* Put modules in `AITemplate/modules/`
* Supports UNet, ControlNet and VAE

[Pre-compiled modules](https://huggingface.co/datasets/hlky/aitemplate)

## TODO

* ~~Mapping VAE depends on AITemplate, specifically AIT_AutoencoderKL, figure out a way to map the parameters without it~~
* ~~Load from LDM models, just needs putting together~~
* ~~More inference examples~~
* ~~Documentation~~ More documentation
* ????

## Supported model types
* ControlNet
* CLIPTextModel
* UNet
* VAE

### WIP Experiments soon:tm:
* x4-upscaler (VAE works)
* DeepFloyd
* Inpaint

### [Compile](https://github.com/hlky/AIT/blob/main/docs/compile.md)

### [CLIP](https://github.com/hlky/AIT/blob/main/docs/clip.md)

### [ControlNet](https://github.com/hlky/AIT/blob/main/docs/controlnet.md)

### [CompVis](https://github.com/hlky/AIT/blob/main/docs/compvis.md)

### [UNet](https://github.com/hlky/AIT/blob/main/docs/unet.md)

### [VAE](https://github.com/hlky/AIT/blob/main/docs/vae.md)
