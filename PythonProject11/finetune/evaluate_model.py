# -*- coding: utf-8 -*-
"""
模型评估脚本
评估微调后的模型性能
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import pandas as pd
from typing import List, Dict
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, base_model_name: str, lora_model_path: str = None,
                 use_quantization: bool = False):
        """
        初始化评估器
        
        Args:
            base_model_name: 基础模型名称
            lora_model_path: LoRA适配器路径（如果使用LoRA）
            use_quantization: 是否使用量化
        """
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.use_quantization = use_quantization
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        logger.info(f"加载基础模型: {self.base_model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        if self.use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        # 如果使用LoRA，加载适配器
        if self.lora_model_path:
            logger.info(f"加载LoRA适配器: {self.lora_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_model_path)
            self.model = self.model.merge_and_unload()  # 合并适配器
        
        self.model.eval()
        logger.info("模型加载完成")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """格式化提示词"""
        if input_text:
            prompt = f"<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n{instruction}\n\n输入：{input_text} [/INST]"
        else:
            prompt = f"<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n{instruction} [/INST]"
        return prompt
    
    def generate(self, prompt: str, max_length: int = 512,
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def evaluate_on_dataset(self, test_data_path: str, output_path: str = None) -> List[Dict]:
        """
        在测试集上评估
        
        Args:
            test_data_path: 测试数据路径
            output_path: 输出结果路径
            
        Returns:
            评估结果列表
        """
        # 加载测试数据
        logger.info(f"加载测试数据: {test_data_path}")
        if test_data_path.endswith('.jsonl'):
            data = []
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        logger.info(f"测试数据量: {len(data)}")
        
        results = []
        
        for item in tqdm(data, desc="评估中"):
            # 提取问题和答案
            text = item.get("text", "")
            # 从text中提取instruction和output（简单解析）
            if "[INST]" in text and "[/INST]" in text:
                parts = text.split("[/INST]")
                if len(parts) >= 2:
                    instruction_part = parts[0].split("[INST]")[-1].strip()
                    ground_truth = parts[1].replace("</s>", "").strip()
                else:
                    instruction_part = text
                    ground_truth = ""
            else:
                instruction_part = text
                ground_truth = ""
            
            # 生成回答
            prompt = self.format_prompt(instruction_part)
            predicted = self.generate(prompt)
            
            result = {
                "instruction": instruction_part[:200],  # 截断用于显示
                "ground_truth": ground_truth[:200],
                "predicted": predicted[:200],
            }
            results.append(result)
        
        # 保存结果
        if output_path:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存到: {output_path}")
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        计算评估指标（简化版，实际可能需要更复杂的指标）
        
        Args:
            results: 评估结果列表
            
        Returns:
            指标字典
        """
        # 这里可以添加更复杂的评估指标，如BLEU、ROUGE等
        # 目前只计算基本的统计信息
        
        total = len(results)
        avg_predicted_length = sum(len(r["predicted"]) for r in results) / total if total > 0 else 0
        
        return {
            "total_samples": total,
            "avg_predicted_length": avg_predicted_length,
        }
    
    def interactive_eval(self):
        """交互式评估"""
        logger.info("进入交互式评估模式（输入'quit'退出）")
        self.load_model()
        
        while True:
            question = input("\n请输入问题: ").strip()
            if question.lower() == 'quit':
                break
            
            if not question:
                continue
            
            prompt = self.format_prompt(question)
            answer = self.generate(prompt)
            print(f"\n回答: {answer}")


def main():
    parser = argparse.ArgumentParser(description="评估微调模型")
    parser.add_argument("--base-model", type=str, required=True,
                       help="基础模型名称或路径")
    parser.add_argument("--lora-model", type=str, default=None,
                       help="LoRA适配器路径")
    parser.add_argument("--test-data", type=str, default=None,
                       help="测试数据路径")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                       help="评估结果输出路径")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式评估模式")
    parser.add_argument("--use-quantization", action="store_true",
                       help="使用4-bit量化")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        base_model_name=args.base_model,
        lora_model_path=args.lora_model,
        use_quantization=args.use_quantization
    )
    
    if args.interactive:
        evaluator.interactive_eval()
    elif args.test_data:
        results = evaluator.evaluate_on_dataset(args.test_data, args.output)
        metrics = evaluator.calculate_metrics(results)
        logger.info(f"评估指标: {metrics}")
    else:
        logger.error("请指定 --test-data 或使用 --interactive 模式")


if __name__ == "__main__":
    main()

