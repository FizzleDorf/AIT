# AIT
 
Experimental usage of [AITemplate](https://github.com/facebookincubator/AITemplate).

[Pre-compiled modules](https://huggingface.co/datasets/hlky/aitemplate)

## ComfyUI custom node

### Installation

This repo can be cloned directly to ComfyUI's custom nodes folder.

Adjust the path as required, the example assumes you are working from the ComfyUI repo.
```
git clone https://github.com/hlky/AIT custom_nodes/AIT
```

### Modules

#### Location

`custom_nodes/AIT/AITemplate/modules/`

#### Compression

Modules are compressed to save bandwidth and storage, refer to the [pre-compiled modules](https://huggingface.co/datasets/hlky/aitemplate) repo for information on the size of decompressed modules.

You can optionally decompress the modules yourself. If a compressed module is selected it is extracted to your OS's temporary directory, these files are cleaned up after each run if the `keep loaded` option is set to `disable`, if `keep loaded` is set to `enable` these files are cleaned the next time you run ComfyUI.

Therefore it is recommended to extract the modules yourself if you will be using `keep loaded` set to `disable`. After the module is extracted in the modules folder, you can safely delete the compressed version and select the decompressed version instead.

#### Nodes

##### Load AITemplate

![image](https://github.com/hlky/AIT/assets/106811348/7242bdf8-96bb-42a1-9e15-3bfeb67beb4e)

![image](https://github.com/hlky/AIT/assets/106811348/6341c23a-f157-417e-b4e4-220a3aca4a5f)

`Loaders > Load AITemplate`

* Supported modules will have `unet` in the name.
* **`control_unet` when using ControlNet.**
* **`inpaint_unet` when using Inpainting models.**

#### Load AITemplate (ControlNet)

![image](https://github.com/hlky/AIT/assets/106811348/1b3991ca-460c-453b-9964-bbefc1a26ba9)

![image](https://github.com/hlky/AIT/assets/106811348/bb64ff88-1b36-442c-9ac2-239134f5e21e)

`Loaders > Load AITemplate (ControlNet)`

* Supported modules will have `controlnet` in the name.
* **Must be used with `control_unet` modules in `Load AITemplate`**

#### VAE Decode (AITemplate)

![image](https://github.com/hlky/AIT/assets/106811348/d273cbc3-a1f2-4e7c-adcd-c194bb03ead0)

![image](https://github.com/hlky/AIT/assets/106811348/1bfec9b3-6a66-4963-8a02-a0cc58739bd7)

`Latent > VAE Decode (AITemplate)`

* Supported modules will have `vae` in the name.
* **not `vae_encode`**

#### VAE Encode (AITemplate)

![image](https://github.com/hlky/AIT/assets/106811348/7b8e26ab-0a91-47b8-a3df-6ddf8f13e9e5)

![image](https://github.com/hlky/AIT/assets/106811348/626d23ff-309c-4273-a0a9-d8e3af129549)

`Latent > VAE Encode (AITemplate)`

* Supported modules will have **`vae_encode`** in the name.
* **not just `vae`**

#### VAE Encode (AITemplate, Inpaint)

![image](https://github.com/hlky/AIT/assets/106811348/e284ef3d-73ba-494d-a133-57c5acf02496)

![image](https://github.com/hlky/AIT/assets/106811348/79fb1a9e-ce7e-4983-a0fd-fed716cd8f8d)

`Latent > Inpaint > VAE Encode (AITemplate)`

* Supported modules will have **`vae_encode`** in the name.
* **not just `vae`**

### Workflow

![image](https://github.com/hlky/AIT/assets/106811348/334c8066-79ea-49d5-be4c-b68f316568e8)

![image](https://github.com/hlky/AIT/assets/106811348/9aa71adc-476a-456d-90e3-60f3e0685801)

![image](https://github.com/hlky/AIT/assets/106811348/ec07169e-0fb8-4a88-a7de-0688050d97c7)

![image](https://github.com/hlky/AIT/assets/106811348/5acb9add-803b-4c6e-a59b-d36e9b5762ce)

![image](https://github.com/hlky/AIT/assets/106811348/5b357f8a-a888-4003-921e-d78df2c8c8cf)

### Errors

* Part of the error will be printed by the AITemplate module so this will be above the trackback.
* If that AITemplate error includes `expected`, `got`, and numbers, you may be using the wrong module, this could be resolution, batch size and you can tell from the numbers, batch size this will be like `1` `2` `4`, for resolution this will be multiples of 64.
* If that AITemplate error includes `constants` you may have selected the wrong kind of module for the node.

## Supported model types
* ControlNet
* CLIPTextModel
* UNet
* **Inpainting UNet**
* VAE
* **VAE encode**

## Developers

### [Compile](https://github.com/hlky/AIT/blob/main/docs/compile.md)

### [CLIP](https://github.com/hlky/AIT/blob/main/docs/clip.md)

### [ControlNet](https://github.com/hlky/AIT/blob/main/docs/controlnet.md)

### [CompVis](https://github.com/hlky/AIT/blob/main/docs/compvis.md)

### [UNet](https://github.com/hlky/AIT/blob/main/docs/unet.md)

### [VAE](https://github.com/hlky/AIT/blob/main/docs/vae.md)
