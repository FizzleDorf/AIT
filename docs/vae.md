# VAE

## Limitations
* None known

## Inference

### Inputs

* `"vae_input"` - vae_input
    * VAE input is the output from UNet
e.g. `torch.randn(batch_size, latent_channels, height, width)` `torch.randn(2, 4, 64, 64)`
or for VAE encode
`torch.randn(batch_size, latent_channels, height, width)` `torch.randn(2, 4, 512, 512)`

### Outputs

`h, w` * `factor`
e.g. `torch.randn(2, 4, 512, 512)`

or for VAE encode

`h, w` // `factor`
e.g. `torch.randn(2, 4, 64, 64)`

## Function

```
def vae_inference(
    exe_module: Model,
    vae_input: torch.Tensor,
    factor: int = 8,
    device: str = "cuda",
    dtype: str = "float16",
    encoder: bool = False,
):
```
`factor` must be set correctly when experimenting with non-standard VAE, default is 8 e.g. `64->512`.
`x4-upscaler` uses `4` as the UNet sample size is bigger e.g. `128->512`.
`factor` is used to set the output shape to accomodate dynamic shape support.
`device` could be specified e.g. `cuda:1` if required
`dtype` is experimental, the module would need to be compiled as `float32`, this is required for `x4-upscaler`.
`encoder` is set for VAE encode inference.
