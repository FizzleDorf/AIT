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
import sys
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from ..modeling.controlnet import (
    ControlNetModel as ait_ControlNetModel,
)
from .util import mark_output

from ait.util.mapping import map_controlnet


def compile_controlnet(
    pt_mod,
    batch_size=(1, 4),
    height=(64, 2048),
    width=(64, 2048),
    clip_chunks=1,
    dim=320,
    hidden_dim=768,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    model_name="ControlNetModel",
    constants=False,
    work_dir="./tmp",
    down_factor=8,
    use_linear_projection=False,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types= (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    in_channels=4,
    out_channels=4,
    sample_size=64,
    class_embed_type=None,
    num_class_embeds=None,
    time_embedding_dim = None,
    conv_in_kernel: int = 3,
    projection_class_embeddings_input_dim = None,
    addition_embed_type = None,
    addition_time_embed_dim = None,
    transformer_layers_per_block = 1,
    dtype="float16",
):
    xl = False
    if projection_class_embeddings_input_dim is not None:
        xl = True
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
    batch_size = batch_size  # double batch size for unet
    ait_mod = ait_ControlNetModel(
        in_channels=in_channels,
        down_block_types=down_block_types,
        block_out_channels=block_out_channels,
        cross_attention_dim=hidden_dim,
        transformer_layers_per_block=transformer_layers_per_block,
        use_linear_projection=use_linear_projection,
        class_embed_type=class_embed_type,
        addition_embed_type=addition_embed_type,
        num_class_embeds=num_class_embeds,
        projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
        dtype="float16",
    )
    ait_mod.name_parameter_tensor()

    pt_mod = pt_mod.eval()
    params_ait = map_controlnet(pt_mod, dim=dim)

    static_shape = width[0] == width[1] and height[0] == height[1] and batch_size[0] == batch_size[1]

    if static_shape:
        batch_size = batch_size[0] * 2  # double batch size for unet
        height_d = height[0] // down_factor
        width_d = width[0] // down_factor
        height_c = height[0]
        width_c = width[0]
        clip_chunks = 77
        embedding_size = clip_chunks
    else:
        batch_size = batch_size[0], batch_size[1] * 2  # double batch size for unet
        batch_size = IntVar(values=list(batch_size), name="batch_size")
        height_d = height[0] // down_factor, height[1] // down_factor
        height_d = IntVar(values=list(height_d), name="height_d")
        width_d = width[0] // down_factor, width[1] // down_factor
        width_d = IntVar(values=list(width_d), name="width_d")
        height_c = height
        height_c = IntVar(values=list(height_c), name="height_c")
        width_c = width
        width_c = IntVar(values=list(width_c), name="width_c")
        clip_chunks = 77, 77 * clip_chunks
        embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, 4], name="latent_model_input", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="timesteps", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim], name="encoder_hidden_states", is_input=True
    )
    controlnet_condition_ait = Tensor(
        [batch_size, height_c, width_c, 3], name="control_hint", is_input=True
    )

    add_embeds = None
    if xl:
        add_embeds = Tensor(
            [batch_size, projection_class_embeddings_input_dim], name="add_embeds", is_input=True, dtype=dtype
        )


    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        text_embeddings_pt_ait,
        controlnet_condition_ait,
        add_embeds=add_embeds,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    dll_name = model_name + ".dll" if sys.platform == "win32" else model_name + ".so"
    compile_model(
        Y, target, work_dir, model_name, constants=params_ait if constants else None, dll_name=dll_name,
    )
