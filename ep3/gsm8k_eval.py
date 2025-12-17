import os
import re
import logging
import json
import pandas as pd
import tqdm
import torch

from megatron.core import mpu
from megatron.training import get_args
from torch import distributed as dist
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.gsm8k_utils import four_shots_prompt, gsm8k_postprocess
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.eval_impl.template import GSM8K_TEMPLATE_DIR

logger = logging.getLogger(__name__)

def clean_number(text):
    """清理数字字符串，移除逗号、美元符号等"""
    if not text:
        return ""
    text = str(text)
    # 移除 LaTeX 包装
    text = text.replace(r'\boxed', '').replace('{', '').replace('}', '')
    # 移除货币和标点
    text = text.replace('$', '').replace(',', '').strip()
    # 移除末尾句号
    if text.endswith('.'):
        text = text[:-1]
    return text

def is_correct(pred, gt):
    """
    比较预测值和真实值是否相等（数值比较）
    """
    pred = clean_number(pred)
    gt = clean_number(gt)

    # 1. 尝试直接字符串匹配
    if pred == gt:
        return True
    
    # 2. 尝试数值匹配 (解决 58.0 == 58 的问题)
    try:
        if float(pred) == float(gt):
            return True
    except ValueError:
        pass
        
    return False

def extract_answer_robust(text):
    """
    鲁棒的答案提取逻辑
    优先级：
    1. \boxed{...} (数学模型常用)
    2. The answer is ... (Prompt引导)
    3. #### ... (GSM8K 格式)
    4. 文本中最后一个数字 (兜底)
    """
    if not text:
        return ""
    
    # 1. 尝试提取 \boxed{content}
    boxed_match = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed_match:
        return boxed_match[-1] # 取最后一个 boxed
    
    # 2. 尝试提取 "The answer is ..."
    template_match = re.findall(r'The answer is (.*?)($|\n|\.)', text)
    if template_match:
        return template_match[-1][0].strip()

    # 3. 尝试提取 #### (如果模型学会了 GSM8K 格式)
    hash_match = text.split('####')
    if len(hash_match) > 1:
        return hash_match[-1].strip()

    # 4. 兜底：提取最后一个数字
    # 匹配整数或浮点数
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    
    return ""

class Gsm8kEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}\n\n{question}",
                 output_template=r'The answer is (.*?) '):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = output_template
        self.batch_size = eval_args.evaluation_batch_size      
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.broadcast_rank = [[0]]
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        final_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        args = get_args()
        
        with open(GSM8K_TEMPLATE_DIR, encoding='utf-8') as f:
            gsm8k_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            with open(file_path, encoding='utf-8') as f:
                gsm8k_list = []
                for line in f.readlines():
                    gsm8k_list.append(json.loads(line))
            
            acc_n = 0
            instructions = []
            answers = [] # 存储 Ground Truth

            if self.max_eval_samples is not None:
                origin_len = len(gsm8k_list)
                gsm8k_list = gsm8k_list[0:min(self.max_eval_samples, origin_len)]
                logger.info("%s length from %s to %s !!!", file_path, str(origin_len), str(len(gsm8k_list)))

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                gsm8k_list, group, align_start_dp_rank = get_final_list_dataset(gsm8k_list, 
                                                                                dist.get_world_size(), 
                                                                                args.tensor_model_parallel_size, 
                                                                                args.pipeline_model_parallel_size)

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(gsm8k_list), desc=file, leave=False)

            index = 0
            for _, item in enumerate(gsm8k_list):
                # 构建 Prompt
                if args.chain_of_thought:
                    instruction = four_shots_prompt + item['question'] + "\nLet's think step by step\nAnswer:"
                else:
                    instruction = self.instruction_template.format(fewshot_template=gsm8k_few_shot_template['few_shot'],
                                                                   question=item['question'])
                instructions.append(instruction)
                
                # --- 提取 Ground Truth ---
                gt_match = re.search(r'####\s*(.*)', item['answer'], re.DOTALL)
                if gt_match:
                    answers.append(gt_match.group(1).strip())
                else:
                    parts = item['answer'].split('####')
                    answers.append(parts[-1].strip() if len(parts) > 1 else "")
                # ------------------------
                
                # 批处理推理
                if len(instructions) == self.batch_size or len(gsm8k_list) == index + 1:
                    chat_results, _ = chat.chat(instruction=instructions, history=[])
                    dist.barrier()
                    
                    if align_start_dp_rank >= 0 and len(gsm8k_list) == index + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]

                    for idx, chat_result in enumerate(chat_results):
                        # 获取模型原始输出
                        raw_output = chat_result[0].lstrip() if isinstance(chat_result, (list, tuple)) else str(chat_result).lstrip()
                        
                        # 提取预测答案
                        pred_answer_str = extract_answer_robust(raw_output)
                        
                        # 记录结果 (为了兼容旧格式，这里还是放列表，但比较时取出来)
                        final_answer_list = [pred_answer_str]
                        
                        try:
                            if dist.get_rank() in group[0]:
                                gt_str = answers[idx]
                                
                                # --- 核心修改：使用数值比较函数 ---
                                is_right = is_correct(pred_answer_str, gt_str)
                                
                                if is_right:
                                    acc_n += 1
                                    logger.info(f"Correct! GT: {gt_str} | Pred: {pred_answer_str}")
                                else:
                                    # 偶尔打印错误日志以便调试
                                    if index % 10 == 0: 
                                        logger.info(f"Wrong. GT: {gt_str} | Pred: {pred_answer_str}")

                        except Exception as e:
                            if dist.get_rank() in group[0]:
                                logger.info(f"Error in comparison: {e}")

                    instructions = []
                    answers = []

                if self.task_pbar is not None:
                    self.task_pbar.update()
                
                index += 1

            # 汇总结果
            if dist.get_rank() in group[0]:
                question_num = len(gsm8k_list)
                if align_start_dp_rank >= 0 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                    question_num -= 1
                
                if not args.broadcast:
                    local_tensor = torch.tensor([acc_n, question_num], device=torch.cuda.current_device())
                    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
                    acc_n, total_questions = local_tensor.tolist()
                else:
                    total_questions = question_num
                
                if dist.get_rank() == 0:
                    acc = acc_n / total_questions if total_questions > 0 else 0
                    logger.info(f'File {file}: {acc_n}/{total_questions} correct, Accuracy: {acc}')
                    total_n += total_questions
                    total_acc_n += acc_n
                    score_datas.append(['gsm8k', total_questions, acc])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if dist.get_rank() in group[0]:
            final_acc = total_acc_n / total_n if total_n > 0 else 0
            logger.info(f"Total GSM8K Accuracy = {total_acc_n}/{total_n} = {final_acc}")
            score_datas.append(["total", total_n, final_acc])
            
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return final_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass