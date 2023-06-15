# ControlNet

## Limitations
* None

## Inference

### Inputs

* `"input0"` - `latent_model_input`
e.g. `torch.randn(batch_size, latent_channels, latent_height, latent_width)` `torch.randn(2, 4, 64, 64)`

* `"input1"` - `timesteps`
e.g. `torch.Tensor([1] * batch_size)`

* `"input2"` - `encoder_hidden_states`
e.g. `torch.randn(batch_size, sequence_length, hidden_dim)` `torch.randn(2, 77, 768)`

* `"input3"` - `controlnet_cond`
    * This is typically the output from a ControlNet annotator.
e.g. `torch.randn(batch_size, control_channels, control_height, control_width)` `torch.randn(2, 3, 512, 512)`

### Outputs

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

## Function

```
def controlnet_inference(
    exe_module: Model,
    latent_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    controlnet_cond: torch.Tensor,
    device: str = "cuda",
    dtype: str = "float16",
):
```
* `device` could be specified e.g. `cuda:1` if required.
* `dtype` is experimental, the module would need to be compiled as `float32`.
