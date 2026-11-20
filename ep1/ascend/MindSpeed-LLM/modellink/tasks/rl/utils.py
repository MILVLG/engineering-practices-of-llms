# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from functools import wraps
from megatron.training import get_args


def get_attr_from_wrapped_model(model, target_attr):
    def recursive_search(module):
        if hasattr(module, target_attr):
            return getattr(module, target_attr)

        for _, child in getattr(module, '_modules').items():
            result = recursive_search(child)
            if result is not None:
                return result

        return None

    return [recursive_search(model)]


def get_tensor_shapes_decorator(get_tensor_shapes):
    @wraps(get_tensor_shapes)
    def wrapper(
            rank,
            model_type,
            seq_length,
            micro_batch_size,
            decoder_seq_length,
            config
    ):
        args = get_args()
        actual_micro_batch_size = getattr(args, "actual_micro_batch_size", None)
        micro_batch_size = micro_batch_size if actual_micro_batch_size is None else actual_micro_batch_size

        tensor_shape = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config
         )

        return tensor_shape

    return wrapper
