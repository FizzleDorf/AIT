#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging

import click
import torch
from aitemplate.testing import detect_target
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from ait.compile.clip import compile_clip

@click.command()
@click.option(
    "--hf-hub-or-path",
    default=r"runwayml/stable-diffusion-v1-5",
    help="the local diffusers pipeline directory or hf hub path e.g. runwayml/stable-diffusion-v1-5",
)
@click.option(
    "--batch-size",
    default=(1, 2),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option(
    "--output-hidden-states",
    default=False,
    type=bool,
    help="Output hidden states",
)
@click.option(
    "--text-projection",
    default=False,
    type=bool,
    help="use text projection",
)
@click.option(
    "--include-constants",
    default=False,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option(
    "--subfolder",
    default="text_encoder",
    help="subfolder of hf repo or path. default `text_encoder`, this is `text_encoder_2` for SDXL.",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--model-name", default="CLIPTextModel", help="module name")
@click.option("--work-dir", default="./tmp", help="work directory")
def compile_diffusers(
    hf_hub_or_path,
    batch_size,
    output_hidden_states,
    text_projection,
    include_constants,
    subfolder="text_encoder",
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    model_name="CLIPTextModel",
    work_dir="./tmp",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    if text_projection:
        pipe = CLIPTextModelWithProjection.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
    else:
        pipe = CLIPTextModel.from_pretrained(
            hf_hub_or_path,
            subfolder=subfolder,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

    compile_clip(
        pipe,
        batch_size=batch_size,
        seqlen=pipe.config.max_position_embeddings,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        output_hidden_states=output_hidden_states,
        text_projection_dim=pipe.config.projection_dim if text_projection else None,
        depth=pipe.config.num_hidden_layers,
        num_heads=pipe.config.num_attention_heads,
        dim=pipe.config.hidden_size,
        act_layer=pipe.config.hidden_act,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
    )

if __name__ == "__main__":
    compile_diffusers()
