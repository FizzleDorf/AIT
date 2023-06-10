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
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.controlnet_unet_2d_condition import (
    ControlNetModel as ait_ControlNetModel,
)
from .util import mark_output

from ait.util.mapping import map_controlnet


def compile_controlnet(
    pt_mod,
    batch_size=1,
    height=512,
    width=512,
    clip_chunks=1,
    dim=320,
    hidden_dim=768,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    model_name="ControlNetModel",
    constants=False,
    work_dir="./tmp"
):
    batch_size = batch_size * 2  # double batch size for unet
    ait_mod = ait_ControlNetModel()
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_controlnet(pt_mod, dim=dim)

    height_d = height // 8
    width_d = width // 8
    height_c = height
    width_c = width

    clip_chunks = 77, 77 * clip_chunks
    embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim], name="input2", is_input=True
    )
    controlnet_condition_ait = Tensor(
        [batch_size, height_c, width_c, 3], name="input3", is_input=True
    )

    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        text_embeddings_pt_ait,
        controlnet_condition_ait,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, work_dir, model_name, constants=params_ait if constants else None
    )
