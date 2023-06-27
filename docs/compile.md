
## Compile

[AITemplate](https://github.com/facebookincubator/AITemplate) must be installed

```
git clone --recursive https://github.com/facebookincubator/AITemplate
cd AITemplate/python
python setup.py bdist_wheel
pip install dist/*.whl
```

### VRAM Usage
* For dynamic shape modules the vram usage of the module will be that of the maximum shape.
* This includes batch size.

### All

* use `--include-constants True` to include the model weights in the compiled module
    * by default the modules do not include model weights
* use `--work-dir` to set the directory where profilers and modules will be built
* use `--model-name` to set the name of the compiled module

### UNet/Control UNet
```
python unet.py --hf-hub-or-path "runwayml/stable-diffusion-v1-5" --width 64 1024 --height 64 1024 --batch-size 1 2 --clip-chunks 8 --model-name "v1_unet_64_1024_1_2" --work-dir "/home/user/ait_tmp/"
```
```
python unet.py --hf-hub-or-path "runwayml/stable-diffusion-v1-5" --width 512 1024 --height 512 1024 --batch-size 1 1 --clip-chunks 8 --model-name "v1_control_unet_512_512" --work-dir "/home/user/ait_tmp/" --controlnet True
```
```
Usage: unet.py [OPTIONS]

Options:
  --hf-hub-or-path TEXT           the local diffusers pipeline directory or hf
                                  hub path e.g. runwayml/stable-diffusion-v1-5
  --width <INTEGER INTEGER>...    Minimum and maximum width
  --height <INTEGER INTEGER>...   Minimum and maximum height
  --batch-size <INTEGER INTEGER>...
                                  Minimum and maximum batch size
  --clip-chunks INTEGER           Maximum number of clip chunks
  --include-constants TEXT        include constants (model weights) with
                                  compiled model
  --use-fp16-acc TEXT             use fp16 accumulation
  --convert-conv-to-gemm TEXT     convert 1x1 conv to gemm
  --controlnet TEXT               UNet for controlnet
  --model-name TEXT               module name
  --work-dir TEXT                 work directory
  --help                          Show this message and exit.
```


### ControlNet
```
python controlnet.py --width 64 1024 --height 64 1024 --batch-size 1 1 --model-name "v1_controlnet_64_512_1" --work-dir "/home/user/ait_tmp/"
```
```
Usage: controlnet.py [OPTIONS]

Options:
  --hf-hub-or-path TEXT           the local diffusers pipeline directory or hf
                                  hub path e.g. lllyasviel/sd-controlnet-canny
  --width <INTEGER INTEGER>...    Minimum and maximum width
  --height <INTEGER INTEGER>...   Minimum and maximum height
  --batch-size <INTEGER INTEGER>...
                                  Minimum and maximum batch size
  --clip-chunks INTEGER           Maximum number of clip chunks
  --include-constants TEXT        include constants (model weights) with
                                  compiled model
  --use-fp16-acc TEXT             use fp16 accumulation
  --convert-conv-to-gemm TEXT     convert 1x1 conv to gemm
  --model-name TEXT               module name
  --work-dir TEXT                 work directory
  --help                          Show this message and exit.
```

### CLIPTextModel
```
python clip.py --hf-hub-or-path "runwayml/stable-diffusion-v1-5" --batch-size 1 8 --model-name "v1_clip_1" --work-dir "/home/user/ait_tmp/"
```
```
Usage: clip.py [OPTIONS]

Options:
  --hf-hub-or-path TEXT           the local diffusers pipeline directory or hf
                                  hub path e.g. runwayml/stable-diffusion-v1-5
  --batch-size <INTEGER INTEGER>...
                                  Minimum and maximum batch size
  --include-constants TEXT        include constants (model weights) with
                                  compiled model
  --use-fp16-acc TEXT             use fp16 accumulation
  --convert-conv-to-gemm TEXT     convert 1x1 conv to gemm
  --model-name TEXT               module name
  --work-dir TEXT                 work directory
  --help                          Show this message and exit.
```

### VAE
```
python vae.py --hf-hub-or-path "runwayml/stable-diffusion-v1-5" --width 64 1024 --height 64 1024 --batch-size 1 1 --model-name "v1_vae_64_1024" --work-dir "/home/user/ait_tmp/"
```
```
Usage: vae.py [OPTIONS]

Options:
  --hf-hub-or-path TEXT           the local diffusers pipeline directory or hf
                                  hub path e.g. runwayml/stable-diffusion-v1-5
  --width <INTEGER INTEGER>...    Minimum and maximum width
  --height <INTEGER INTEGER>...   Minimum and maximum height
  --batch-size <INTEGER INTEGER>...
                                  Minimum and maximum batch size
  --fp32 TEXT                     use fp32
  --include-constants TEXT        include constants (model weights) with
                                  compiled model
  --use-fp16-acc TEXT             use fp16 accumulation
  --convert-conv-to-gemm TEXT     convert 1x1 conv to gemm
  --model-name TEXT               module name
  --work-dir TEXT                 work directory
  --help                          Show this message and exit.
```
