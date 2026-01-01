# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    def __init__(self, 
                 base_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 output_dir: str = "models/finetuned",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: list = None):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self, use_quantization: bool = False):
        logger.info(f"加载模型: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        if use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self, data_path: str, split: str = "train"):
        dataset = load_dataset("json", data_files={split: data_path}, split=split)
        logger.info(f"加载了 {len(dataset)} 条数据")
        return dataset
    
    def tokenize_function(self, examples):
        texts = examples["text"]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def train(self,
              train_data_path: str,
              val_data_path: str = None,
              num_epochs: int = 3,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              save_steps: int = 500,
              logging_steps: int = 50,
              use_quantization: bool = False):
        self.load_model_and_tokenizer(use_quantization)
        
        train_dataset = self.load_dataset(train_data_path, "train")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = None
        if val_data_path:
            eval_dataset = self.load_dataset(val_data_path, "validation")
            eval_dataset = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("开始训练...")
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        lora_path = self.output_dir / "lora_adapter"
        lora_path.mkdir(exist_ok=True)
        self.model.save_pretrained(str(lora_path))
        
        logger.info("训练完成")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                       default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--use-quantization", action="store_true")
    
    args = parser.parse_args()
    
    trainer = LoRATrainer(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    trainer.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_quantization=args.use_quantization
    )


if __name__ == "__main__":
    main()

