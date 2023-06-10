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
from diffusers import UNet2DConditionModel
from ait.compile.unet import compile_unet

@click.command()
@click.option(
    "--hf-hub-or-path",
    default="./tmp/diffusers-pipeline/runwayml/stable-diffusion-v1-5",
    help="the local diffusers pipeline directory or hf hub path e.g. runwayml/stable-diffusion-v1-5",
)
@click.option(
    "--width",
    default=(64, 2048),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum width",
)
@click.option(
    "--height",
    default=(64, 2048),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum height",
)
@click.option(
    "--batch-size",
    default=(1, 4),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option("--clip-chunks", default=6, help="Maximum number of clip chunks")
@click.option(
    "--include-constants",
    default=None,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--controlnet", default=False, help="UNet for controlnet")
@click.option("--model-name", default="UNet2DConditionModel", help="module name")
@click.option("--work-dir", default="./tmp", help="work directory")
def compile_diffusers(
    hf_hub_or_path,
    width,
    height,
    batch_size,
    clip_chunks,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    controlnet=False,
    model_name="UNet2DConditionModel",
    work_dir="./tmp",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    assert (
        width[0] % 64 == 0 and width[1] % 64 == 0
    ), "Minimum Width and Maximum Width must be multiples of 64, otherwise, the compilation process will fail."
    assert (
        height[0] % 64 == 0 and height[1] % 64 == 0
    ), "Minimum Height and Maximum Height must be multiples of 64, otherwise, the compilation process will fail."

    pipe = UNet2DConditionModel.from_pretrained(
        hf_hub_or_path,
        subfolder="unet" if not hf_hub_or_path.endswith("unet") else None,
        torch_dtype=torch.float16,
    ).to("cuda")

    compile_unet(
       pipe,
       batch_size=batch_size,
       width=width,
       height=height,
       clip_chunks=clip_chunks,
       use_fp16_acc=use_fp16_acc,
       convert_conv_to_gemm=convert_conv_to_gemm,
       hidden_dim=pipe.config.cross_attention_dim,
       attention_head_dim=pipe.config.attention_head_dim,
       use_linear_projection=pipe.config.get("use_linear_projection", False),
       block_out_channels=pipe.config.block_out_channels,
       down_block_types=pipe.config.down_block_types,
       up_block_types=pipe.config.up_block_types,
       in_channels=pipe.config.in_channels,
       out_channels=pipe.config.out_channels,
       class_embed_type=pipe.config.class_embed_type,
       num_class_embeds=pipe.config.num_class_embeds,
       only_cross_attention=pipe.config.only_cross_attention,
       dim=pipe.config.block_out_channels[0],
       constants=True if include_constants else False,
       controlnet=True if controlnet else False,
       model_name=model_name,
       work_dir=work_dir,
    )

if __name__ == "__main__":
    compile_diffusers()
