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

import os
import abc
import logging
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.training import get_args, global_vars
from megatron.core import parallel_state

logger = logging.getLogger(__name__)

from modellink.tasks.preprocess.templates import Template, get_model_template


class MegatronModuleForCausalLMABC(torch.nn.Module, abc.ABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self):
        super(MegatronModuleForCausalLMABC, self).__init__()
        self.top_k = 50
        self.top_p = 1.0
        self.do_sample = False
        self.num_beams = 1
        self.temperature = 1.0
        self.max_length = 128
        self.max_new_tokens = 0
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None
        self.num_return_sequences = 1
        self.length_penalty = 1.0
        self.tokenizer_new = None
        self.recompute = True
        self.detokenize = True
        self.include_input = False
        self.stream = False
        self.return_output_log_probs = False

    @classmethod
    def from_pretrained(
            cls,
            model_provider,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ):
        """
        This is an API for initializing model and loading weight.

        Parameters:
        ----------
        model_provider(`func`):
            Function used to generate model objects which is similar to the training define.
        pretrained_model_name_or_path(`str`, *optional*, defaults to None):
           File path of Model weight in megatron format (TP, PP may be used).
           If it is None, the random initialized weights will be used.
        """

    def generate(self, input_ids=None, **kwargs):
        """
        This is an API for text generation which complies with most huggingface definition.

        - *greedy decoding* if `do_sample=False`
        - *top-k decoding* if `top_k>0`
        - *top-p decoding* if `top_p>0.0`
        - *beam-search decoding* if `num_beams>1`

        Parameters:
        ----------
        input_ids(str | list | LongTensor):
            The text entered by the user, e.g. 'Hello!'
            Or
            The text, which encoded by tokenizer, entered by the user, e.g. [0, 13, 5, ...]
            Or
            The List, which includes multi texts or tokens,
            e.g. [['Hello!'], ["How are you?"]] | [[0, 13, 5, ...], [0, 21, ...]].
            Notice that in beam-search mode multi texts or tokens is forbidden.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        top_k (`int`, *optional*, defaults to 0):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token. Optionally,
            use a list to set multiple *end-of-sequence* tokens.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token. Optionally,
            use a list to set multiple *beginning-of-sequence* tokens.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        tokenizer (`obj`, *optional*, defaults to None):
            If you don't want to use the tokenizer initialized by megatron, you can pass it in HF format here.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences. Only activate in beam search mode.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch. Only activate
            in beam search mode.
        recompute (`bool`, *optional*, defaults to True):
            Whether the model not to uses the last result in computing next token.
        detokenize (`bool`, *optional*, defaults to True):
            Whether to detokenize tokens into characters.
        include_input (`bool`, *optional*, defaults to False):
            Whether the output contains the context instruction.
        stream (`bool`, *optional*, defaults to False):
            Whether the output is streamed one by one.
        return_output_log_probs(`bool`, *optional*, defaults to False):
            Whether to return a probability distribution for each token.
            Note that the accumulated probability (i.e. Score) of the whole sentence will be return in beam search mode.
        """
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_length = kwargs.pop("max_length", 128)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 0)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.tokenizer_new = kwargs.pop("tokenizer", None)
        self.recompute = kwargs.pop("recompute", True)
        self.detokenize = kwargs.pop("detokenize", True)
        self.include_input = kwargs.pop("include_input", False)
        self.stream = kwargs.pop("stream", False)
        self.return_output_log_probs = kwargs.pop("return_output_log_probs", False)


class MegatronModuleForCausalLM(MegatronModuleForCausalLMABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self, *args, **kwargs):
        super(MegatronModuleForCausalLM, self).__init__()
        from megatron.training import get_tokenizer
        from .utils import greedy_search_or_sampling
        from .generation import beam_search
        from .communication import broadcast_float_list

        args = get_args()
        args.max_tokens_to_oom = args.max_tokens_to_oom if hasattr(args, "max_tokens_to_oom") else 4096
        args.inference_batch_times_seqlen_threshold = args.inference_batch_times_seqlen_threshold \
            if hasattr(args, "inference_batch_times_seqlen_threshold") else 4

        self.padded_vocab_size = args.padded_vocab_size
        self.pipeline_size_larger_than_one = args.pipeline_model_parallel_size > 1

        try:
            self.tokenizer = get_tokenizer().tokenizer
        except AssertionError:
            self.tokenizer = None

        # import module to avoid error of circular import
        self.greedy_search_or_sampling = greedy_search_or_sampling
        self.beam_search_in_sampling = beam_search
        self.broadcast_float_list = broadcast_float_list
        self.template = None
        if hasattr(args, "prompt_type") and args.prompt_type is not None:
            self.template = get_model_template(args.prompt_type.strip())

    @staticmethod
    def _ids_check(ids, tokenizer):
        checked_ids = []
        for per_ids in ids:
            if per_ids == torch.Size([]) and torch.max(per_ids) >= len(tokenizer):
                warning_info = "The output ids exceeds the tokenizer length, " \
                               "the clamp operation is enforced, please check!!"
                logging.warning(warning_info)
                checked_ids.append(torch.clamp(per_ids, min=0, max=len(tokenizer)) - 1)
            else:
                checked_ids.append(per_ids)
        return checked_ids

    @classmethod
    def from_pretrained(
            cls,
            model_provider, pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ) -> MegatronModuleForCausalLMABC:
        from megatron.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from megatron.core.distributed import DistributedDataParallel as LocalDDP
        from megatron.legacy.model import Float16Module as MegatronFloat16Module
        from megatron.training.utils import unwrap_model

        args = get_args()

        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        args.model = get_model(model_provider, wrap_with_ddp=False)

        if pretrained_model_name_or_path:
            args.load = pretrained_model_name_or_path

        if args.load:
            load_checkpoint(args.model, None, None, 'load')

        unwrap_classes = (torchDDP, LocalDDP, MegatronFloat16Module)

        return unwrap_model(args.model, unwrap_classes)[0]

    def generate(self, input_ids=None, **kwargs):
        args = get_args()

        if parallel_state.get_data_parallel_world_size() // parallel_state.get_expert_model_parallel_world_size() > 1:
            raise ValueError("In this inference mode data parallel is forbidden.")

        super(MegatronModuleForCausalLM, self).generate(input_ids=input_ids, **kwargs)

        # =======================================
        # Make sure input params are available
        # to all ranks
        # =======================================
        self._broadcast_config(args)

        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)
        stop_ids = []
        if hasattr(args, "add_eos_token") and args.add_eos_token:
            for token in args.add_eos_token:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if len(ids) == 1:
                    stop_ids.append(ids[0])
                else:
                    logger.warning("[DEBUG] Unable to add stop token '%s' (encoded ids=%s)", token, ids)
            if getattr(args, "debug_print_eval_batch", False):
                logger.info("[DEBUG] extra stop_ids=%s", stop_ids)
        # =======================================
        # Tokenize the prompts and broadcasting,
        # so you don't need to pass the prompt on
        # each process.
        # =======================================
        context_tokens, master_rank = self._tokenize(input_ids)
        args.master_rank = master_rank
        args.micro_batch_size = len(context_tokens)
        if getattr(args, "debug_print_eval_batch", False):
            token_lens = [len(tokens) for tokens in context_tokens]
            logger.info("[DEBUG] Megatron.generate micro_batch=%s token_lens(sample)=%s",
                        args.micro_batch_size,
                        token_lens[:min(8, len(token_lens))])

        stop_token = [args.eos_id] + stop_ids

        if hasattr(args, "prompt_type") and args.prompt_type is not None:
            stop_ids = stop_ids + [self.tokenizer.convert_tokens_to_ids(token) for token in self.template.stop_words] + \
                       [self.tokenizer.eos_token_id]

            stop_token = [args.eos_id] + stop_ids

        # =======================================
        # Get the streaming tokens generator
        # =======================================
        if self.num_beams > 1:
            token_stream = self.beam_search_in_sampling(
                args.model[0],
                context_tokens,
                beam_size=self.num_beams,
                stop_token=stop_token,
                num_return_gen=self.num_return_sequences,
                length_penalty=self.length_penalty
            )
        else:
            token_stream = self.greedy_search_or_sampling(
                args.model[0],
                context_tokens,
                stop_ids=stop_ids,
            )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        return self._token_generator(token_stream)

    def _broadcast_config(self, args):
        values = [
            self.num_beams,
            self.do_sample,
            self.top_k,
            self.top_p,
            self.temperature,
            self.max_length,
            self.max_new_tokens,
            self.length_penalty,
            self.return_output_log_probs,
            self.stream
        ]

        values_float_tensor = self.broadcast_float_list(len(values), float_list=values)
        self.num_beams = int(values_float_tensor[0].item())
        self.do_sample = bool(values_float_tensor[1].item())
        self.top_k = int(values_float_tensor[2].item())
        self.top_p = values_float_tensor[3].item()
        self.temperature = values_float_tensor[4].item()
        self.max_length = int(values_float_tensor[5].item())
        self.max_new_tokens = int(values_float_tensor[6].item())
        self.length_penalty = values_float_tensor[7].item()
        self.return_output_log_probs = bool(values_float_tensor[8].item())
        self.stream = bool(values_float_tensor[9].item())

        setattr(args, "text_generation_config", {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "temperature": self.temperature,
            "recompute": self.recompute,
            "return_output_log_probs": self.return_output_log_probs,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "greedy": True if not self.do_sample else False
        })

    def _init_tokenizer(self, args):
        self.tokenizer = self.tokenizer if self.tokenizer_new is None else self.tokenizer_new
        global_vars._GLOBAL_TOKENIZER = self.tokenizer

        if self.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.pad_token_id
        if self.eos_token_id is not None:
            self.tokenizer.eos_token_id = self.eos_token_id
        if self.bos_token_id is not None:
            self.tokenizer.bos_token_id = self.bos_token_id

        if self.tokenizer.eos_token_id is not None:
            args.eos_id = self.tokenizer.eos_token_id
            args.eod_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Your tokenizer doesn't include eos_token.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def _encode_no_template(self, input_ids):
        context_tokens = [[]]
        if isinstance(input_ids, str):
            context_tokens = [self.tokenizer.encode(input_ids)]
        elif torch.is_tensor(input_ids):
            if len(input_ids.shape) == 1:
                context_tokens = input_ids.unsqueeze(0).numpy().tolist()
            elif len(input_ids.shape) == 2:
                context_tokens = input_ids.numpy().tolist()
        elif isinstance(input_ids, (tuple, list)):
            if len(input_ids) and isinstance(input_ids[0], (tuple, list)):
                context_tokens = input_ids
            elif len(input_ids) and isinstance(input_ids[0], int):
                context_tokens = [input_ids]
            elif len(input_ids) and isinstance(input_ids[0], str):
                context_tokens = [self.tokenizer.encode(val) for val in input_ids]
        else:
            raise TypeError("Please check input_ids in correct type.")

        return context_tokens


    def _encode_by_template(self, input_ids):
        context_tokens = []

        if input_ids is None:
            return [[]]
        response_prompt = [{"role": "assistant", "content": ""}]
        if len(input_ids) and isinstance(input_ids, str):
            paired_messages = [{"role": "user", "content": "{}".format(input_ids)}] + response_prompt
            tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
            context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (dict)):
            paired_messages = input_ids + response_prompt
            tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
            context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (str)):
            for query in input_ids:
                paired_messages = [{"role": "user", "content": "{}".format(query)}] + response_prompt
                tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
                context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (tuple, list)):
            for val in input_ids:
                if len(val) and isinstance(val, (tuple, list)):
                    paired_messages = val + response_prompt
                    tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
                    context_tokens.append(tokens)
        else:
            raise TypeError("Please check input_ids in correct type.")


        return context_tokens if len(context_tokens) > 0 else [context_tokens]


    def _tokenize(self, input_ids):
        context_tokens = [[]]
        broadcast_rank = torch.zeros(dist.get_world_size(),
                                     dtype=torch.int64,
                                     device=torch.device(torch.cuda.current_device()))
        if input_ids is not None and len(input_ids) > 0:
            args = get_args()
            if args.hf_chat_template:
                if not hasattr(self.tokenizer, "apply_chat_template"):
                    raise AssertionError('The tokenizer has no Huggingface chat template, Please use chat model.')

                context_tokens = [self.tokenizer.apply_chat_template(
                    input_ids,
                    tokenize=True,
                    add_generation_prompt=True
                )]
            elif self.template is None:
                context_tokens = self._encode_no_template(input_ids)
            else:
                context_tokens = self._encode_by_template(input_ids)


            broadcast_rank[dist.get_rank()] = 1

        dist.all_reduce(broadcast_rank)
        master_rank = torch.nonzero(broadcast_rank)[0, 0]

        return context_tokens, master_rank

    def _post_processing(self, output, context_lengths, log_probs):
        if not self.include_input:
            output = [val[context_lengths[i]:] for i, val in enumerate(output)]

        # When batch size > 1, you need truncate the tokens after eos_token_id
        self._truncate_in_multi_batch(output)

        if self.detokenize:
            try:
                output_checked = self._ids_check(output, self.tokenizer)
                output = self.tokenizer.batch_decode(output_checked, skip_special_tokens=True)
            except Exception as e:
                error_info = "Meet errors when trying to decode the tokens. "\
                             "Please handle it by yourself."
                logging.error(error_info)
                logging.error(e)

        output = output[0] if len(output) == 1 else output

        if not self.return_output_log_probs:
            res = output
        else:
            if self.num_beams == 1:
                log_probs = [val[context_lengths[i]:, :] for i, val in enumerate(log_probs)] \
                    if log_probs is not None else None

            res = output, log_probs[0] if len(log_probs) == 1 else log_probs

        return res

    def _truncate_in_multi_batch(self, output):
        if len(output) > 1:
            for idx, batch in enumerate(output):
                trunc_index = torch.nonzero(batch == self.tokenizer.eos_token_id)

                if min(trunc_index.shape):
                    output[idx][trunc_index.min():] = self.tokenizer.eos_token_id

    def _yield(self, token_stream):
        output, context_lengths, log_probs = None, None, None
        for output, context_lengths, log_probs in token_stream:
            if self.stream:
                res = self._post_processing(output, context_lengths, log_probs)
                yield res

        if not self.stream:
            yield self._post_processing(output, context_lengths, log_probs)

    def _token_generator(self, token_stream):
        token_stream = self._yield(token_stream)
        if not self.stream:
            full_output = None
            for tmp in token_stream:
                full_output = tmp
            return full_output
        else:
            return token_stream
        
        
class GPTModelInfer(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_model = MegatronModuleForCausalLM()

    def generate(self, input_ids=None, **kwargs):
        return self.infer_model.generate(input_ids=input_ids, **kwargs)
