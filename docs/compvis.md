# CompVis

## AITemplate Model Wrapper

* Used in place of `CFGNoisePredictor` which is then wrapped by `CompVisDenoiser`/`CompVisVDenoiser`
* Provides `apply_model`
```
def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict,
    ):
```
* `timesteps_pt` = `t`
* `latent_model_input` = `x`
* `cond` supports `c_crossattn`, `c_concat`
* for ControlNet, additional residuals are expected under `control`
    * `down_block_residuals` = `control`.`output`
    * `mid_block_residual` = `control`.`middle[0]`

Input is passed to [`unet_inference`](https://github.com/hlky/AIT/blob/main/docs/unet.md)
