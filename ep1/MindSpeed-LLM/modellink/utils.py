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

"""General utilities."""
import os
import stat
import random
from functools import wraps
import torch
import torch_npu
from torch import distributed as dist
import numpy as np
import megatron
from megatron.training import get_args
from megatron.core import mpu
from tqdm import tqdm
from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                             get_context_parallel_for_hybrid_ulysses_rank,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank)


WRITE_FILE_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_FILE_DEFAULT_MODES = stat.S_IWUSR | stat.S_IRUSR

_ACTUAL_SEQ_LEN = None


def get_actual_seq_len():
    return _ACTUAL_SEQ_LEN


def set_actual_seq_len(actual_seq_len):
    global _ACTUAL_SEQ_LEN
    _ACTUAL_SEQ_LEN = actual_seq_len


def compute_actual_seq_len(seq):
    zero_pos = (seq == 0).nonzero()[1:].squeeze(dim=1)
    res = zero_pos.tolist()
    res.append(len(seq))
    return res


def generate_actual_seq_len(batch):
    position_ids = batch['position_ids'].view(-1)
    actual_seq_len = compute_actual_seq_len(position_ids)
    set_actual_seq_len(actual_seq_len)


def parse_args():
    return megatron.training.arguments.parse_args()


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
                torch.distributed.get_rank() % torch.cuda.device_count() == 0
        ):
            return True
        else:
            return False
    else:
        return True


def print_rank0_by_args(args, message):
    """Before initialization of distributed, we only print on rank 0."""
    if args.rank == 0:
        print(message, flush=True)


def get_tune_attention_mask(attention_mask_1d):
    args = get_args()
    micro_batch_size, seq_length = attention_mask_1d.size()
    if args.reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None

    if args.tokenizer_padding_side == "left":
        attention_mask = torch.tril(torch.ones(seq_length, seq_length, device=attention_mask_1d.device, dtype=torch.bool)).view(1, 1, seq_length, seq_length)
        attention_mask_tran = attention_mask_1d.view(seq_length, 1, -1)
        attention_mask = attention_mask.masked_fill((attention_mask_tran < 0.5).view(-1, 1, 1, seq_length), value=0)
    else:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def print_args_wrapper(fn):
    """
    Add switch for controlling when to print arguments.
    """
    @wraps(fn)
    def wrapper(title, args, after_validate=False):
        if after_validate:
            fn(title, args)

    return wrapper


def print_args(title, args):
    """
    Provide a public func for printing arguments.
    """
    # here global process group has not been initialized, that's why we use args.rank
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def emit(self, record):
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1 # 如果获取rank失败，则设置为一个不合法的rank

    if rank == 0 or rank == -1:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_device_wrapper(fn):
    @wraps(fn)
    def wrapper(local_rank=None, *arg, **kwargs):
        backend = torch.distributed.get_backend()
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('npu')
            else:
                device = torch.device(f'npu:{local_rank}')
        else:
            device = fn(local_rank)
        return device

    return wrapper


def unwrap_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model, module_instances=None):
        if not module_instances:
            module_instances = megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES
        return fn(model, module_instances)
    return wrapper


def get_finetune_data_on_this_tp_rank(data_iterator):
    args = get_args()
    ds = next(data_iterator)
    tokens = ds.get('input_ids').long().cuda(non_blocking=True)
    tokens_shape = tokens.shape
    micro_batch_size = tokens_shape[0]

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        via_length = torch.LongTensor([tokens_shape[1]]).cuda(non_blocking=True)
        _broadcast(via_length)
        _broadcast(tokens)
        attention_mask_1d = ds.get('attention_mask').long().cuda(non_blocking=True)
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d)
    else:
        via_length = torch.empty((1), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(via_length)
        tokens = torch.empty((micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(tokens)
        attention_mask_1d = torch.empty((micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(attention_mask_1d)
        attention_mask = get_tune_attention_mask(attention_mask_1d)

    return tokens, attention_mask


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            'position_ids': data["position_ids"].cuda(non_blocking=True)
        }
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            if args.reset_position_ids:
                _broadcast(batch['position_ids'])
        else:
            _broadcast(batch['attention_mask'])
            if args.reset_position_ids:
                _broadcast(batch['position_ids'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            if args.reset_position_ids:
                _broadcast(position_ids)
            else:
                position_ids = None

        else:
            tokens = None
            labels = None
            loss_mask = None
            _broadcast(attention_mask)
            if args.reset_position_ids:
                _broadcast(position_ids)
            else:
                position_ids = None

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()
    cp_size = args.context_parallel_size
    if not cp_size > 1:
        return batch

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            if len(val.shape) != 2:
                raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
            seq_dim = 0
            mask_row = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=1)]
            batch[key] = mask_list
            continue
        if val is not None:
            seq_dim = 1
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * r_size,
                val.shape[seq_dim] // (2 * r_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            if len(val.shape) != 2:
                raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
            seq_dim = 0
            mask_row = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
            mask_list = [m.contiguous() for m in mask_row.chunk(r_size, dim=1)]
            batch[key] = mask_list
            continue

        if val is not None:
            seq_dim = 1
            val = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
            batch[key] = val

    return batch
