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
    default="./tmp/diffusers-pipeline/lllyasviel/sd-controlnet-canny",
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
@click.option("--model-name", default="ControlNetModel", help="module name")
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
    model_name="ControlNetModel",
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

    pipe = ControlNetModel.from_pretrained(
        hf_hub_or_path,
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
    )

if __name__ == "__main__":
    compile_diffusers()
