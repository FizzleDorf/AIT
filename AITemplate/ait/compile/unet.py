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
    ControlNetUNet2DConditionModel as ait_ControlNetUNet2DConditionModel,
)
from ..modeling.unet_2d_condition import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import mark_output
from ait.util.mapping import map_unet

def compile_unet(
    pt_mod,
    batch_size=(1, 8),
    height=(64, 2048),
    width=(64, 2048),
    clip_chunks=1,
    work_dir="./tmp",
    dim=320,
    hidden_dim=1024,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    controlnet=False,
    x4_upscaler=False,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
    model_name="UNet2DConditionModel",
    use_linear_projection=False,
    constants=True,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types= (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    in_channels=4,
    out_channels=4,
    sample_size=64,
    class_embed_type=None,
    num_class_embeds=None,
    only_cross_attention=[
        True,
        True,
        True,
        False
    ],
    down_factor=8,
):
    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(block_out_channels)

    if controlnet:
        model_name = "ControlNet" + model_name
        ait_mod = ait_ControlNetUNet2DConditionModel(
            sample_size=sample_size,
            cross_attention_dim=hidden_dim,
            attention_head_dim=attention_head_dim,
            use_linear_projection=use_linear_projection,
        )
    else:
        ait_mod = ait_UNet2DConditionModel(
            sample_size=sample_size,
            cross_attention_dim=hidden_dim,
            attention_head_dim=attention_head_dim,
            use_linear_projection=use_linear_projection,
            up_block_types=up_block_types,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            only_cross_attention=only_cross_attention,
        )
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pt_mod.eval()
    params_ait = map_unet(pt_mod, dim=dim)

    static_shape = (width[0] == width[1] and height[0] == height[1] and batch_size[0] == batch_size[1]) or controlnet

    if static_shape:
        batch_size = batch_size[0] * 2  # double batch size for unet
        height = height[0] // down_factor
        width = width[0] // down_factor
        height_d = height
        width_d = width
    else:
        batch_size = batch_size[0], batch_size[1] * 2  # double batch size for unet
        batch_size = IntVar(values=list(batch_size), name="batch_size")
        height = height[0] // down_factor, height[1] // down_factor
        width = width[0] // down_factor, width[1] // down_factor
        height_d = IntVar(values=list(height), name="height_d")
        width_d = IntVar(values=list(width), name="width_d")

    if static_shape:
        embedding_size = 77
    else:
        clip_chunks = 77, 77 * clip_chunks
        embedding_size = IntVar(values=list(clip_chunks), name="embedding_size")
        

    latent_model_input_ait = Tensor(
        [batch_size, height_d, width_d, in_channels], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor(
        [batch_size, embedding_size, hidden_dim], name="input2", is_input=True
    )

    class_labels = None
    if x4_upscaler:
        class_labels = Tensor(
            [batch_size], name="input3", dtype="int64", is_input=True
        )

    mid_block_additional_residual = None
    down_block_additional_residuals = None
    if controlnet:
        down_block_residual_0 = Tensor(
            [batch_size, height, width, block_out_channels[0]],
            name="down_block_residual_0",
            is_input=True,
        )
        down_block_residual_1 = Tensor(
            [batch_size, height, width, block_out_channels[0]],
            name="down_block_residual_1",
            is_input=True,
        )
        down_block_residual_2 = Tensor(
            [batch_size, height, width, block_out_channels[0]],
            name="down_block_residual_2",
            is_input=True,
        )
        down_block_residual_3 = Tensor(
            [batch_size, height // 2, width // 2, block_out_channels[0]],
            name="down_block_residual_3",
            is_input=True,
        )
        down_block_residual_4 = Tensor(
            [batch_size, height // 2, width // 2, block_out_channels[1]],
            name="down_block_residual_4",
            is_input=True,
        )
        down_block_residual_5 = Tensor(
            [batch_size, height // 2, width // 2, block_out_channels[1]],
            name="down_block_residual_5",
            is_input=True,
        )
        down_block_residual_6 = Tensor(
            [batch_size, height // 4, width // 4, block_out_channels[1]],
            name="down_block_residual_6",
            is_input=True,
        )
        down_block_residual_7 = Tensor(
            [batch_size, height // 4, width // 4, block_out_channels[2]],
            name="down_block_residual_7",
            is_input=True,
        )
        down_block_residual_8 = Tensor(
            [batch_size, height // 4, width // 4, block_out_channels[2]],
            name="down_block_residual_8",
            is_input=True,
        )
        down_block_residual_9 = Tensor(
            [batch_size, height // 8, width // 8, block_out_channels[2]],
            name="down_block_residual_9",
            is_input=True,
        )
        down_block_residual_10 = Tensor(
            [batch_size, height // 8, width // 8, block_out_channels[3]],
            name="down_block_residual_10",
            is_input=True,
        )
        down_block_residual_11 = Tensor(
            [batch_size, height // 8, width // 8, block_out_channels[3]],
            name="down_block_residual_11",
            is_input=True,
        )
        mid_block_residual = Tensor(
            [batch_size, height // 8, width // 8, block_out_channels[3]],
            name="mid_block_residual",
            is_input=True,
        )


    if controlnet:
        Y = ait_mod(
            latent_model_input_ait,
            timesteps_ait,
            text_embeddings_pt_ait,
            down_block_residual_0,
            down_block_residual_1,
            down_block_residual_2,
            down_block_residual_3,
            down_block_residual_4,
            down_block_residual_5,
            down_block_residual_6,
            down_block_residual_7,
            down_block_residual_8,
            down_block_residual_9,
            down_block_residual_10,
            down_block_residual_11,
            mid_block_residual,
        )
    else:
        Y = ait_mod(
            latent_model_input_ait,
            timesteps_ait,
            text_embeddings_pt_ait,
            class_labels,
            mid_block_additional_residual,
            down_block_additional_residuals,
        )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y, target, work_dir, model_name, constants=params_ait if constants else None
    )
