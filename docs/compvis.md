# CompVis

## AITemplate Model Wrapper

* Used in place of `CFGNoisePredictor` which is then wrapped by `CompVisDenoiser`/`CompVisVDenoiser`
* Provides `apply_model`
```
def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_crossattn = None,
        c_concat = None,
        control = None,
        transformer_options = None,
    ):
```
* `timesteps_pt` = `t`
* `latent_model_input` = `x`
* `c_crossattn` = encoder_hidden_states
* `c_concat` = will be concatenated to `latent_model_input` if not `None`
* for ControlNet, additional residuals are expected under `control`
    * `down_block_residuals` = `control`.`output`
    * `mid_block_residual` = `control`.`middle[0]`
* `transformer_options` is unused but present for ComfyUI compatibility

Input is passed to [`unet_inference`](https://github.com/hlky/AIT/blob/main/docs/unet.md)
