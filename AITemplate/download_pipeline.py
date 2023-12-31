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
import click
import torch
try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError(
        "Please install diffusers with `pip install diffusers` to use this script."
    )


@click.command()
@click.option("--token", default="", help="access token")
@click.option(
    "--hf-hub",
    default="runwayml/stable-diffusion-v1-5",
    help="hf hub",
)
@click.option(
    "--save_directory",
    default="./tmp/diffusers-pipeline/runwayml/stable-diffusion-v1-5",
    help="pipeline files local directory",
)
@click.option("--revision", default="fp16", help="access token")
def download_pipeline_files(token, hf_hub, save_directory, revision="fp16") -> None:
    StableDiffusionPipeline.from_pretrained(
        hf_hub,
        revision=revision if revision != "" else None,
        torch_dtype=torch.float16,
        # use provided token or the one generated with `huggingface-cli login``
        use_auth_token=token if token != "" else True,
    ).save_pretrained(save_directory)


if __name__ == "__main__":
    download_pipeline_files()
