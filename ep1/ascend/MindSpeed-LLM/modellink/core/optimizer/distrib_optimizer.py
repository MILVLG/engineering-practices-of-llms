# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

import itertools
from functools import wraps
from typing import Callable, Dict, List, Optional
import torch.distributed
from apex.optimizers import FusedAdam as Adam

from megatron.training import get_args
from megatron.core.distributed import ParamAndGradBuffer
from megatron.core.optimizer.grad_scaler import MegatronGradScaler
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer


def distributed_optimizer_init(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        per_model_buffers: Dict[int, List[ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: torch.distributed.ProcessGroup,
        data_parallel_group_idx: int,
):
    MixedPrecisionOptimizer.__init__(
        self, optimizer, config, grad_scaler, init_state_fn,
    )

    assert isinstance(
        optimizer, Adam
    ), "Only Adam currently supported, due to checkpointing requirements."

    # Model grad buffer ranges.
    assert per_model_buffers is not None, "per_model_buffers must be provided"
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    self.per_model_buffers = per_model_buffers
    self.data_parallel_group = data_parallel_group
    self.data_parallel_group_gloo = data_parallel_group_gloo
    self.data_parallel_group_idx = data_parallel_group_idx
    self.gbuf_idx_to_model_idx_map = {}
    gbuf_idx = 0
    for model_idx, buffers in self.per_model_buffers.items():
        for _ in buffers:
            self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
            gbuf_idx += 1
    self.gbuf_ranges = []
    self.per_bucket_numel = []
    self.per_bucket_numel_unpadded = []
    for buffer in self.buffers:

        self.per_bucket_numel.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.grad_data.numel() for bucket in buffer.buckets
                ]
            }
        )
        self.per_bucket_numel_unpadded.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.numel_unpadded for bucket in buffer.buckets
                ]
            }
        )
        self.gbuf_ranges.append(self._build_gbuf_range_map(buffer, self.data_parallel_group))
    self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

    # Optimizer ranges.
    (
        self.model_param_group_index_map,
        self.opt_group_ranges,
    ) = self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self._build_model_and_main_param_groups(
        self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
    )

    # Now construct data structures to manage all-gather handles.
    self.all_gather_handles = []
    self.all_gather_handle_index_to_bucket_index_map = []
    self.model_index_to_all_gather_handle_index_map = {}
    self.all_gather_handle_indices = []
    self.param_to_all_gather_handle_index_map = {}

    self.pbuf_view_items = self._get_model_param_buffer_dp_views()
    for (gbuf_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
        self.all_gather_handle_index_to_bucket_index_map.append(
            (gbuf_index, dtype, bucket_index)
        )
        all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
        self.all_gather_handles.append(None)

        # Store all all_gather_handle_indices.
        model_idx = self.gbuf_idx_to_model_idx_map[gbuf_index]
        if model_idx not in self.model_index_to_all_gather_handle_index_map:
            self.model_index_to_all_gather_handle_index_map[model_idx] = []
        self.model_index_to_all_gather_handle_index_map[model_idx].append(
            all_gather_handle_index
        )

        for param in self.buffers[gbuf_index].buckets[bucket_index].params_list:
            self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
    self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

    self.overlap_param_gather = self.config.overlap_param_gather
    self.remove_pre_hook_handle = None
    if self.overlap_param_gather:
        self.enable_pre_hook()

    self.update_successful = False

    self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())


def distributed_optimizer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        argument = get_args()
        if argument.enable_high_availability:
            distributed_optimizer_init(self, *args, **kwargs)
        else:
            fn(self, *args, **kwargs)
    return wrapper