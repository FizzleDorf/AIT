# CLIP

## Limitations
* None known

## Inference

### Inputs

* sequence_length is typically 77
* longer prompts are accepted by UNet modules when compiled with appropiate `--clip-chunks` option which provides maximum prompt length `clip_chunks * 77`

* `"input0"` - `input_ids`
```
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_input = tokenizer(
    ["a photo of an astronaut riding a horse on mars"] * batch_size,
    padding="max_length",
    max_length=sequence_length,
    truncation=True,
    return_tensors="pt",
)
input_ids = text_input["input_ids"].cuda()
```

* `"input1"` - position_ids
    * dealt with internally by `clip_inference`
e.g. `torch.arange(sequence_length).expand((batch, -1)).to(device)`

### Outputs

`torch.randn(batch_size, sequence_length, hidden_dim)` `torch.randn(1, 77, 768)`

## Function

```
def clip_inference(
    exe_module: Model,
    input_ids: torch.Tensor,
    seqlen: int = 77,
    device: str = "cuda",
    dtype: str = "float16",
):
```
* `seqlen` is unlikely to need to be changed.
* `device` could be specified e.g. `cuda:1` if required
* `dtype` is experimental, the module would need to be compiled as `float32`
