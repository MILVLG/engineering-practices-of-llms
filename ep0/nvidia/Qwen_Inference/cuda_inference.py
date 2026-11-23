"""
NVIDIA GPU (CUDA) inference implementation for ep0 Qwen_Inference.

This module provides a CUDA-native counterpart to the original Ascend NPU
inference code (`exp2.ipynb`, `utils/session.py`, `utils/inference.py`),
without modifying any original files.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}. Choose from {list(mapping.keys())}.")
    return mapping[key]


@dataclass
class CudaInferenceConfig:
    hf_model_dir: str
    device_str: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    sampling_method: str = "greedy"  # greedy | top_k | top_p
    sampling_value: Optional[float] = None
    temperature: float = 1.0
    max_input_length: int = 1024
    max_output_length: int = 2048
    attn_implementation: str = "sdpa"  # sdpa | eager

    def __post_init__(self) -> None:
        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no NVIDIA GPU is available.")
        self.device = torch.device(self.device_str)
        self.dtype = _resolve_dtype(self.torch_dtype)
        # Fallback from bfloat16 to float16 on GPUs without native bf16 support
        if self.device.type == "cuda" and self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.dtype = torch.float16
        self.tokenizer_dir = self.hf_model_dir


class CudaSession:
    """
    Manages a HuggingFace CausalLM on NVIDIA CUDA GPU with incremental KV-cache.
    Mirrors the original `Session` interface (`run`, `reset`).
    """

    def __init__(self, config: CudaInferenceConfig) -> None:
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.run_times = 0
        self.past_key_values = DynamicCache()

        print(f"[INFO] CudaSession loading model from {config.hf_model_dir}")
        print(f"[INFO] Device: {self.device} ({torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'}), dtype: {self.dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_dir,
            torch_dtype=self.dtype,
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def run(self, input_ids: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Run one prefill (seq_len > 1) or decode (seq_len == 1) step and return
        the last-position logits as a float32 numpy array of shape (1, 1, vocab_size).
        """
        if isinstance(input_ids, np.ndarray):
            input_ids_t = torch.from_numpy(input_ids).to(device=self.device, dtype=torch.long)
        else:
            input_ids_t = input_ids.to(device=self.device, dtype=torch.long)

        if input_ids_t.ndim == 1:
            input_ids_t = input_ids_t.unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids_t,
            past_key_values=self.past_key_values,
            use_cache=True,
        )
        self.past_key_values = outputs.past_key_values
        self.run_times += input_ids_t.shape[-1]

        # Only transfer the last token logits to CPU for sampling
        last_logits = outputs.logits[:, -1:, :].float().cpu().numpy()
        return last_logits

    def reset(self) -> None:
        self.run_times = 0
        self.past_key_values = DynamicCache()


class CudaInference:
    """
    NVIDIA GPU inference engine compatible with the ep0 Qwen_Inference workflow.
    Supports greedy, top-k, and top-p sampling, multi-turn chat history, and streaming.
    """

    def __init__(self, config: CudaInferenceConfig) -> None:
        self.config = config
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length
        self.device = config.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_dir,
            trust_remote_code=True,
        )
        self.session = CudaSession(config)
        self.state: Dict[str, object] = {"code": 200, "isEnd": False, "message": ""}
        self.lock = Lock()
        self.first = True
        self.reset()
        print("[INFO] CUDA inference init success")

    @staticmethod
    def sample_logits(
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: Optional[float] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Numerically stable token sampling supporting greedy, top_k, and top_p.
        `logits` shape: (1, vocab_size).
        """
        if temperature <= 0 or sampling_method == "greedy":
            return np.argmax(logits, axis=-1).astype(np.int64)

        if sampling_method in ("top_k", "top_p"):
            if sampling_value is None:
                raise ValueError(f"sampling_value must be provided for {sampling_method}")
            scaled = logits.astype(np.float64) / float(temperature)
            scaled -= np.max(scaled, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)

            if sampling_method == "top_k":
                k = max(1, min(int(sampling_value), sorted_probs.shape[-1]))
                cutoff = k
            else:  # top_p
                p = float(sampling_value)
                cumulative_probs = np.cumsum(sorted_probs[0])
                idx = int(np.searchsorted(cumulative_probs, p))
                cutoff = min(idx + 1, sorted_probs.shape[-1])

            probs_of_interest = sorted_probs[0, :cutoff].copy()
            indices_of_interest = sorted_indices[0, :cutoff]
            probs_of_interest /= np.sum(probs_of_interest)
            chosen = np.random.choice(indices_of_interest, p=probs_of_interest)
            return np.array([chosen], dtype=np.int64)

        raise ValueError(f"Unknown sampling method: {sampling_method}")

    def reset(self) -> None:
        self.first = True
        self.session.reset()

    def getState(self) -> Dict[str, object]:
        with self.lock:
            return self.state.copy()

    def _build_input_ids(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]],
        system_prompt: str,
    ) -> torch.Tensor:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for user_msg, bot_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer([text], return_tensors="pt")["input_ids"].to(torch.long)
        return input_ids[:, -self.max_input_length :]

    def predict(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1024,
        sampling_method: str = "greedy",
        sampling_value: Optional[float] = None,
        temperature: float = 1.0,
        return_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """Run end-to-end autoregressive generation on NVIDIA GPU."""
        self.reset()
        with self.lock:
            self.state["isEnd"] = False
            self.state["message"] = ""

        input_ids = self._build_input_ids(prompt, history, system_prompt)
        input_length = input_ids.shape[1]
        max_output_len = min(self.max_output_length - input_length, max_new_tokens)

        ids_list: List[int] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()
        t_first_token = t_start

        for step in range(max_output_len):
            logits = self.session.run(input_ids)
            next_token = self.sample_logits(
                logits[0],
                sampling_method=sampling_method,
                sampling_value=sampling_value,
                temperature=temperature,
            )
            token_id = int(next_token[0])
            if step == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t_first_token = time.perf_counter()

            ids_list.append(token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            input_ids = next_token.reshape(1, 1)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()

        text_out = self.tokenizer.decode(ids_list, skip_special_tokens=False)
        with self.lock:
            self.state["message"] = text_out
            self.state["isEnd"] = True

        if return_stats:
            num_gen = len(ids_list)
            ttft = t_first_token - t_start
            decode_time = max(t_end - t_first_token, 1e-6)
            decode_tps = (num_gen - 1) / decode_time if num_gen > 1 else 0.0
            stats = {
                "input_tokens": input_length,
                "output_tokens": num_gen,
                "ttft_s": ttft,
                "total_s": t_end - t_start,
                "decode_tok_per_s": decode_tps,
                "vram_peak_gb": (
                    torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
            return text_out, stats

        return text_out

    def stream_predict(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1024,
        sampling_method: str = "greedy",
        sampling_value: Optional[float] = None,
        temperature: float = 1.0,
    ) -> Generator[str, None, None]:
        """Yield incremental text chunks as tokens are generated."""
        self.reset()
        input_ids = self._build_input_ids(prompt, history, system_prompt)
        max_output_len = min(self.max_output_length - input_ids.shape[1], max_new_tokens)

        ids_list: List[int] = []
        printed_len = 0
        for _ in range(max_output_len):
            logits = self.session.run(input_ids)
            next_token = self.sample_logits(
                logits[0],
                sampling_method=sampling_method,
                sampling_value=sampling_value,
                temperature=temperature,
            )
            token_id = int(next_token[0])
            if token_id == self.tokenizer.eos_token_id:
                break
            ids_list.append(token_id)
            text_out = self.tokenizer.decode(ids_list, skip_special_tokens=True)
            new_text = text_out[printed_len:]
            if "\ufffd" in new_text:
                input_ids = next_token.reshape(1, 1)
                continue
            if new_text:
                yield new_text
                printed_len = len(text_out)
            input_ids = next_token.reshape(1, 1)

    def close(self) -> None:
        if hasattr(self, "session"):
            self.session.reset()
            if hasattr(self.session, "model"):
                del self.session.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
        print("[INFO] CUDA inference resources released")


def run_ep0_benchmarks(
    model_dir: str,
    device_str: str = "cuda:0",
    torch_dtype: str = "bfloat16",
) -> None:
    """Run all 7 experiment prompts from ep0/Qwen_Inference/exp2.ipynb on NVIDIA GPU."""
    config = CudaInferenceConfig(
        hf_model_dir=model_dir,
        device_str=device_str,
        torch_dtype=torch_dtype,
        max_input_length=1024,
        max_output_length=2048,
    )
    inference = CudaInference(config)

    cases = [
        (
            "1. Basic Prediction",
            "What is the capital of France?",
            dict(sampling_method="greedy", max_new_tokens=128),
        ),
        (
            "2. Greedy Sampling",
            "What is the capital of France?",
            dict(sampling_method="greedy", max_new_tokens=128),
        ),
        (
            "3. Top-K Sampling (k=5)",
            "What are the benefits of artificial intelligence?",
            dict(sampling_method="top_k", sampling_value=5, temperature=0.8, max_new_tokens=256),
        ),
        (
            "4. Top-P Sampling (p=0.9)",
            "Can you explain quantum computing?",
            dict(sampling_method="top_p", sampling_value=0.9, temperature=0.8, max_new_tokens=256),
        ),
        (
            "5. Short Input/Output (max_new_tokens=64)",
            "What is 2 + 2?",
            dict(sampling_method="greedy", max_new_tokens=64),
        ),
        (
            "6. Long Input/Output (max_new_tokens=64)",
            "Can you explain how quantum entanglement works and its potential applications in future technologies?",
            dict(sampling_method="greedy", max_new_tokens=64),
        ),
        (
            "7. Extreme Length Input/Output (max_new_tokens=128)",
            "Can you give me a detailed explanation of machine learning algorithms, including supervised learning, unsupervised learning, and reinforcement learning, and provide examples of each?",
            dict(sampling_method="greedy", max_new_tokens=128),
        ),
    ]

    try:
        for title, prompt, kwargs in cases:
            print("=" * 72)
            print(f"[CASE] {title}")
            print(f"[PROMPT] {prompt}")
            output, stats = inference.predict(prompt, return_stats=True, **kwargs)
            print(f"[OUTPUT]\n{output}")
            print(
                f"[STATS] in={stats['input_tokens']} tok | out={stats['output_tokens']} tok | "
                f"TTFT={stats['ttft_s']*1000:.1f} ms | decode={stats['decode_tok_per_s']:.1f} tok/s | "
                f"total={stats['total_s']:.2f} s | peak_vram={stats['vram_peak_gb']:.2f} GB"
            )
    finally:
        inference.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ep0 Qwen Inference on NVIDIA CUDA GPU")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "Qwen2.5-3B-Instruct"),
        help="Local path or HuggingFace repo ID for Qwen2/Qwen2.5 Instruct model",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device string (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Precision: bfloat16 | float16 | float32")
    args = parser.parse_args()
    run_ep0_benchmarks(model_dir=args.model_dir, device_str=args.device, torch_dtype=args.dtype)


if __name__ == "__main__":
    main()
