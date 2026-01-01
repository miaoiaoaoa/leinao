# -*- coding: utf-8 -*-
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, output_dir: str = "data/finetune"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_llama2_prompt(self, instruction: str, input_text: str = "", 
                            response: str = "") -> str:
        if input_text:
            prompt = f"<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n{instruction}\n\n输入：{input_text} [/INST] {response} </s>"
        else:
            prompt = f"<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n{instruction} [/INST] {response} </s>"
        return prompt
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
        return data
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
        return data
    
    def load_csv_data(self, file_path: str, 
                     instruction_col: str = "instruction",
                     input_col: Optional[str] = None,
                     output_col: str = "output") -> List[Dict]:
        df = pd.read_csv(file_path)
        data = []
        for _, row in df.iterrows():
            item = {
                "instruction": row[instruction_col],
                "output": row[output_col]
            }
            if input_col and input_col in df.columns:
                item["input"] = row[input_col]
            data.append(item)
        logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
        return data
    
    def convert_to_training_format(self, data: List[Dict], 
                                   format_type: str = "alpaca") -> List[Dict]:
        formatted_data = []
        
        for item in data:
            if format_type == "alpaca":
                instruction = item.get("instruction", item.get("question", ""))
                input_text = item.get("input", item.get("context", ""))
                output = item.get("output", item.get("answer", item.get("response", "")))
                
                prompt = self.format_llama2_prompt(instruction, input_text, output)
                formatted_data.append({
                    "text": prompt
                })
                
            elif format_type == "qa":
                question = item.get("question", item.get("instruction", ""))
                answer = item.get("answer", item.get("output", item.get("response", "")))
                
                prompt = self.format_llama2_prompt(question, "", answer)
                formatted_data.append({
                    "text": prompt
                })
        
        logger.info(f"转换了 {len(formatted_data)} 条数据到训练格式")
        return formatted_data
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.9,
                   val_ratio: float = 0.05, test_ratio: float = 0.05) -> Dict[str, List[Dict]]:
        import random
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            "train": data[:train_end],
            "validation": data[train_end:val_end],
            "test": data[val_end:]
        }
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"保存了 {len(data)} 条数据到 {file_path}")
    
    def process(self, input_file: str, input_format: str = "json",
               format_type: str = "alpaca", output_name: str = "train",
               split: bool = True):
        if input_format == "json":
            data = self.load_json_data(input_file)
        elif input_format == "jsonl":
            data = self.load_jsonl_data(input_file)
        elif input_format == "csv":
            data = self.load_csv_data(input_file)
        else:
            raise ValueError(f"不支持的数据格式: {input_format}")
        
        formatted_data = self.convert_to_training_format(data, format_type)
        
        if split:
            splits = self.split_data(formatted_data)
            for split_name, split_data in splits.items():
                output_file = self.output_dir / f"{output_name}_{split_name}.jsonl"
                self.save_jsonl(split_data, output_file)
        else:
            output_file = self.output_dir / f"{output_name}.jsonl"
            self.save_jsonl(formatted_data, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input-format", type=str, default="json", 
                       choices=["json", "jsonl", "csv"])
    parser.add_argument("--format-type", type=str, default="alpaca",
                       choices=["alpaca", "qa"])
    parser.add_argument("--output-dir", type=str, default="data/finetune")
    parser.add_argument("--output-name", type=str, default="train")
    parser.add_argument("--no-split", action="store_true")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.output_dir)
    preprocessor.process(
        input_file=args.input,
        input_format=args.input_format,
        format_type=args.format_type,
        output_name=args.output_name,
        split=not args.no_split
    )
    
    logger.info("数据准备完成")


if __name__ == "__main__":
    main()

