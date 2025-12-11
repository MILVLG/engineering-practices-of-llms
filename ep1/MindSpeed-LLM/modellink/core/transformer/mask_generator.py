# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
import torch
from megatron.training import get_args

_MASK_CACHE = None
FA_COMPRESSED_MASK_LEN = 2048
MUST_COMPRESS = 'compressed'


def _do_generate_atten_mask(should_compress: bool = True,
                            seq_length: int = FA_COMPRESSED_MASK_LEN, pre_tockens=None, next_tockens=None):
    current_device = torch.cuda.current_device()
    valid_seq_length = seq_length is not None and seq_length >= 1
    valid_token_config = pre_tockens is not None and next_tockens is not None
    if should_compress:
        return torch.triu(
            torch.ones([FA_COMPRESSED_MASK_LEN, FA_COMPRESSED_MASK_LEN], dtype=bool, device=current_device), diagonal=1)
    elif valid_seq_length:
        if not valid_token_config:
            # default for casual atten mask
            return torch.triu(torch.ones([seq_length, seq_length], dtype=bool, device=current_device), diagonal=1)
        mask1 = torch.tril(torch.ones([seq_length, seq_length], device=current_device), diagonal=-(pre_tockens + 1))
        mask2 = torch.triu(torch.ones([seq_length, seq_length], device=current_device), diagonal=next_tockens + 1)
        return (mask1 + mask2).bool().npu()
    else:
        raise ValueError(f"failed to build attention mask for seq_length={seq_length}, pre_tockens={pre_tockens}, next_tockens={next_tockens}.")


def set_attention_mask(attn_mask):
    """
    Currently only for UT ST test.
    """
    global _MASK_CACHE
    _MASK_CACHE = attn_mask


def get_attention_mask(mode=None):
    """
    Generate mask and cache it to boost mask generation.
    Variable sequence length is unsupported.
    """
    global _MASK_CACHE
    # set params of FA
    args = get_args()

    if args.cp_attention_mask_type == 'full' or args.variable_seq_lengths:
        return None

    if _MASK_CACHE is None:
        # For seq_length > 2k, mode 4 would have memory gain.
        is_ring_attention = args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']
        should_compress = args.use_flash_attn and args.seq_length > FA_COMPRESSED_MASK_LEN
        # mode == MUST_COMPRESS would override should_compress.
        should_compress = mode == MUST_COMPRESS or should_compress or is_ring_attention
        args.sparse_mode = 4 if should_compress else 0
        # casual attention
        _MASK_CACHE = _do_generate_atten_mask(should_compress=should_compress, seq_length=args.seq_length,
                                              pre_tockens=args.pre_tockens, next_tockens=args.next_tockens)

    return _MASK_CACHE
