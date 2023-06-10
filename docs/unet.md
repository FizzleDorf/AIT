# UNet

## Limitations
* Control UNet only supports static shape.
* Models with odd input channels (Inpaint, x4-upscaler etc.) are experimental.

## Inference

* AITemplate uses `bhwc`
* for input to `_inference` functions provide input as `bchw`, output will be in `bchw`

### Inputs

* `"input0"` - `latent_model_input`
e.g. `torch.randn(batch_size, latent_channels, height, width)` `torch.randn(2, 4, 64, 64)`

* `"input1"` - `timesteps`
e.g. `torch.Tensor([1] * batch_size)`

* `"input2"` - `encoder_hidden_states`
e.g. `torch.randn(batch_size, sequence_length, hidden_dim)` `torch.randn(2, 77, 768)`

#### ControlNet

These are the output from ControlNet modules. The sizes are determined by batch size, latent height and width, and block_out_channels.

* `"down_block_residual_{i}"` i = 0..11
e.g.
```
torch.Size([2, 64, 64, 320])
torch.Size([2, 64, 64, 320])
torch.Size([2, 64, 64, 320])
torch.Size([2, 32, 32, 320])
torch.Size([2, 32, 32, 640])
torch.Size([2, 32, 32, 640])
torch.Size([2, 16, 16, 640])
torch.Size([2, 16, 16, 1280])
torch.Size([2, 16, 16, 1280])
torch.Size([2, 8, 8, 1280])
torch.Size([2, 8, 8, 1280])
torch.Size([2, 8, 8, 1280])
```

* `"mid_block_residual"`
e.g.
`torch.Size([2, 8, 8, 1280])`

#### Experimental

* `"input3"` - `class_labels`
    * This is the noise level for `x4-upscaler`
e.g. `torch.tensor([20] * batch_size, dtype=torch.long)`

### Outputs

Same size as `latent_model_input` e.g. `torch.randn(batch_size, latent_channels, height, width)` `torch.randn(2, 4, 64, 64)`

## Function

```
def unet_inference(
    exe_module: Model,
    latent_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    class_labels: torch.Tensor = None,
    down_block_residuals: List[torch.Tensor] = None,
    mid_block_residual: torch.Tensor = None,
    device: str = "cuda",
    dtype: str = "float16",
):
```
* `class_labels` is experimental and used by `x4-upscaler`.
* `down_block_residuals` and `mid_block_residual` require a `Control UNet` module.
* `device` could be specified e.g. `cuda:1` if required.
* `dtype` is experimental, the module would need to be compiled as `float32`.
