# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import types
import importlib
from functools import wraps
import torch
import apex
from mindspeed.optimizer.adamw import AdamW
from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine


def mock_megatron_dependencies():
    mock_transformer_engine()
    mock_amp_c()
    mock_flash_attn()
    mock_fused_layer_norm_cuda()


def mock_transformer_engine():
    def version_wrapper(fn):
        @wraps(fn)
        def wrapper(name, *args, **kwargs):
            if name == 'transformer-engine':
                return '0.0'
            res = fn(name, *args, **kwargs)
            return res

        return wrapper
    importlib.metadata.version = version_wrapper(importlib.metadata.version)
    sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
    sys.modules['transformer_engine'].__spec__ = 'te'
    setattr(sys.modules['transformer_engine'], 'pytorch', torch.nn)
    setattr(sys.modules['transformer_engine'].pytorch, 'LayerNormLinear', torch.nn.Module)
    setattr(sys.modules['transformer_engine'].pytorch, 'DotProductAttention', torch.nn.Module)


def mock_amp_c():
    def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
        total_norm = 0.0
        norm_type = 2.0
        ret_per_tensor = [] if per_parameter else None
        for grads_for_norm in tensor_lists:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type
            if per_parameter:
                ret_per_tensor.append(total_norm.clone())
        if not tensor_lists:
            grad_norm = torch.cuda.FloatTensor([0])
            total_norm = grad_norm ** norm_type
        return total_norm ** (1 / norm_type), ret_per_tensor


    def multi_tensor_scale(overflow_buf, tensor_lists, scale):
        if len(tensor_lists) != 2:
            raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
        if len(tensor_lists[0]) != len(tensor_lists[1]):
            raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                                len(tensor_lists[1])))

        with torch.no_grad():
            for i in range(len(tensor_lists[0])):
                tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)

    sys.modules['amp_C'] = types.ModuleType('amp_C')
    setattr(sys.modules['amp_C'], 'multi_tensor_l2norm', multi_tensor_l2norm)
    setattr(sys.modules['amp_C'], 'multi_tensor_scale', multi_tensor_scale)


def mock_flash_attn():
    sys.modules['flash_attn.flash_attn_interface'] = types.ModuleType('flash_attn_flash_attn_interface')
    setattr(sys.modules['flash_attn.flash_attn_interface'], 'flash_attn_unpadded_func', torch.nn.Module)


def mock_fused_layer_norm_cuda():
    sys.modules['fused_layer_norm_cuda'] = types.ModuleType('fused_layer_norm_cuda')


def patch_npu_apex_torch():
    patch_npu_torch()
    patch_npu_apex()


def patch_npu_apex():

    def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
        return op(noop_flag_buffer, tensor_lists, *args)

    apex.optimizers.FusedAdam = AdamW  # replace apex fused adam
    apex.multi_tensor_apply.multi_tensor_applier = multi_tensor_applier
    apex.normalization.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine


def patch_npu_torch():
    def type_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            if isinstance(res, str):
                res = res.replace('npu', 'cuda')
            return res
        return wrapper

    # Patch view method to ensure tensor is contiguous before performing view
    def ensure_contiguous(fn):
        def wrapper(tensor, *args, **kwargs):
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return fn(tensor, *args, **kwargs)

        return wrapper

    def repeat_interleave(inputs, repeats, dim):
        shape = inputs.shape
        new_shape = shape[:dim + 1] + (repeats,) + shape[dim + 1:]
        out_shape = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
        return inputs.unsqueeze(dim + 1).expand(new_shape).reshape(out_shape)

    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)  # replace npu type to gpu type
    torch.Tensor.type = type_wrapper(torch.Tensor.type)  # replace npu type to gpu type
    torch.Tensor.view = ensure_contiguous(torch.Tensor.view)  # patch view to ensure tensor is contiguous
    torch.Tensor.repeat_interleave = repeat_interleave # replace npu implementation of torch.repeat_interleave
    torch.compile = torch.jit.script