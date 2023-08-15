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

try:
    from diffusers import ControlNetModel
except ImportError:
    raise ImportError(
        "Please install diffusers with `pip install diffusers` to use this script."
    )
from ait.compile.controlnet import compile_controlnet


@click.command()
@click.option(
    "--hf-hub-or-path",
    default="lllyasviel/sd-controlnet-canny",
    help="the local diffusers pipeline directory or hf hub path e.g. lllyasviel/sd-controlnet-canny",
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
    default=(1, 1),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option("--clip-chunks", default=10, help="Maximum number of clip chunks")
@click.option(
    "--include-constants",
    default=None,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--model-name", default="ControlNetModel", help="module name")
@click.option("--work-dir", default="./tmp", help="work directory")
@click.option("--out-dir", default="./out", help="out directory")
def compile_diffusers(
    hf_hub_or_path,
    width,
    height,
    batch_size,
    clip_chunks,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    model_name="ControlNetModel",
    work_dir="./tmp",
    out_dir="./out",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    pipe = ControlNetModel.from_pretrained(
        hf_hub_or_path,
        use_safetensors=True,
        # variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    compile_controlnet(
        pipe,
        batch_size=batch_size,
        width=width,
        height=height,
        clip_chunks=clip_chunks,
        convert_conv_to_gemm=convert_conv_to_gemm,
        use_fp16_acc=use_fp16_acc,
        constants=include_constants,
        model_name=model_name,
        work_dir=work_dir,
        hidden_dim=pipe.config.cross_attention_dim,
        use_linear_projection=pipe.config.get("use_linear_projection", False),
        block_out_channels=pipe.config.block_out_channels,
        down_block_types=pipe.config.down_block_types,
        in_channels=pipe.config.in_channels,
        class_embed_type=pipe.config.class_embed_type,
        num_class_embeds=pipe.config.num_class_embeds,
        dim=pipe.config.block_out_channels[0],
        time_embedding_dim=None,
        projection_class_embeddings_input_dim=pipe.config.projection_class_embeddings_input_dim
        if hasattr(pipe.config, "projection_class_embeddings_input_dim")
        else None,
        addition_embed_type=pipe.config.addition_embed_type
        if hasattr(pipe.config, "addition_embed_type")
        else None,
        addition_time_embed_dim=pipe.config.addition_time_embed_dim
        if hasattr(pipe.config, "addition_time_embed_dim")
        else None,
        transformer_layers_per_block=pipe.config.transformer_layers_per_block
        if hasattr(pipe.config, "transformer_layers_per_block")
        else 1,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    compile_diffusers()
