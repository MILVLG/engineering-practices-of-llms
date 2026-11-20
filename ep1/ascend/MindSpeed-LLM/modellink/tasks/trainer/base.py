# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import sys
import time
from functools import partial
from typing import Union

import torch

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.checkpointing import save_checkpoint
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.training import (
    print_datetime,
    get_one_logger,
    append_to_progress_log,
    evaluate_and_print_results
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from modellink.initialize import set_jit_fusion_options
from modellink.training import train
from modellink.training import build_train_args
from modellink.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank, generate_actual_seq_len
from modellink.tasks.trainer.utils import train_valid_test_datasets_provider


_TRAIN_START_TIME = time.time()


class BaseTrainer:
    """
    A basic trainer that provides fundamental functions such as get_batch, forward_step, loss_func, etc.
    for regular PyTorch and Softmax training. For reinforcement learning scenarios,
    some of these methods may need to be overridden.
    """
    def __init__(self, process_non_loss_data_func=None):
        self.args = None
        self.timers = None
        self.train_args = None
        self.model_type = ModelType.encoder_or_decoder
        self.test_data_iterator_list = None
        self.train_valid_test_datasets_provider = None
        self.process_non_loss_data_func = process_non_loss_data_func

        self.initialize()

    def initialize(self):
        train_valid_test_datasets_provider.is_distributed = True

        self.args = get_args()
        self.timers = get_timers()
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider

        if self.args.log_progress:
            append_to_progress_log("Starting job")
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()
        # Adjust the startup time, so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor(
            [_TRAIN_START_TIME],
            dtype=torch.float,
            device='cuda'
        )
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        one_logger = get_one_logger()
        if one_logger:
            one_logger.log_metrics({
                'train_iterations_warmup': 5
            })

        train_args, test_data_iterator_list = build_train_args(
            self.args,
            self.timers,
            self.train_valid_test_datasets_provider,
            self.model_provider,
            self.model_type,
            self.forward_step,
            self.process_non_loss_data_func
        )
        self.train_args = train_args
        self.test_data_iterator_list = test_data_iterator_list

    @staticmethod
    def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
        """Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
            Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if args.use_mcore_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                        args.moe_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )
        else:
            if not args.context_parallel_size == 1:
                raise ValueError("Context parallelism is only supported with Megatron Core!")

            model = megatron.legacy.model.GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )

        return model

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""

        args = get_args()

        if args.is_instruction_dataset:
            if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
                if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                    tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)

                    return tokens, None, None, attention_mask, None
                else:
                    return None, None, None, None, None
            # Items and their type.
            keys = ['input_ids', 'attention_mask', 'labels']
            data_type = torch.int64

            # Broadcast data.
            data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

            # Unpack
            labels = data_b.get('labels').long()
            tokens = data_b.get('input_ids').long()
            attention_mask_1d = data_b.get('attention_mask').long()
            # ignored label -100
            loss_mask = torch.where(labels == -100, 0, 1)

            attention_mask = get_tune_attention_mask(attention_mask_1d)

            return tokens, labels, loss_mask, attention_mask, None

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)
        if args.reset_position_ids:
            generate_actual_seq_len(batch)
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

        return batch.values()

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        loss_mask = input_tensor

        losses = output_tensor.float()
        if self.args.is_instruction_dataset:
            loss_mask = loss_mask[..., 1:].view(-1).float()
        else:
            loss_mask = loss_mask.view(-1).float()
        if self.args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss[0] / loss[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if self.args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model: GPTModel):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

        return output_tensor, partial(self.loss_func, loss_mask)

    def train(self):
        args = get_args()
        model_provider = self.model_provider
        model_type = self.model_type
        forward_step_func = self.forward_step
        test_data_iterator_list = self.test_data_iterator_list
        train_args = self.train_args
        train_valid_test_dataset_provider = self.train_valid_test_datasets_provider
        process_non_loss_data_func = None

        if not args.skip_train:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_project_dir:
                assert args.retro_cyclic_train_iters is not None
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                if args.enable_high_availability:
                    try:
                        from mindio_ttp.adaptor import tft_init_controller_processor, tft_register_processor, tft_train
                    except ModuleNotFoundError:
                        sys.exit("The mindio_ttp package is not installed. Exiting.")
                    tft_init_controller_processor(enable_tls=False, tls_option_top_path='')
                    tft_register_processor(train_valid_test_dataset_provider, model_provider, model_type)
                    iteration, num_floating_point_operations_so_far = tft_train(train_args, test_data_iterator_list)
                else:
                    iteration, num_floating_point_operations_so_far = train(*train_args)

                test_data_iterator = test_data_iterator_list[0]
                (
                    forward_step_func,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    train_data_iterator,
                    valid_data_iterator,
                    process_non_loss_data_func,
                    config
                ) = train_args

            print_datetime('after training is done')

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                save_checkpoint(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far
                )
        else:
            print_rank_0('skipping training (--skip-train is on) ...')

            iteration = args.iteration

        if args.do_valid:
            prefix = f'iteration {iteration} on validation set'
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func, config,
                                       verbose=True, write_to_tensorboard=not args.skip_train)

        if args.do_test:
            prefix = f'iteration {iteration} on test set'
            evaluate_and_print_results(prefix, forward_step_func,
                                       test_data_iterator, model,
                                       iteration, process_non_loss_data_func, config,
                                       verbose=True, write_to_tensorboard=not args.skip_train)
