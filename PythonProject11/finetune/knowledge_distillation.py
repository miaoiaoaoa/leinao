# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import logging
from pathlib import Path
import argparse
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationTrainer:
    def __init__(self,
                 teacher_model_path: str,
                 student_model_path: str,
                 temperature: float = 5.0,
                 alpha: float = 0.5,
                 use_quantization: bool = False):
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.temperature = temperature
        self.alpha = alpha
        self.use_quantization = use_quantization
        
        self.teacher_model = None
        self.student_model = None
        self.teacher_tokenizer = None
        self.student_tokenizer = None
    
    def load_models(self):
        logger.info("加载教师模型...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        teacher_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_path,
            **teacher_kwargs
        )
        self.teacher_model.eval()
        logger.info("教师模型加载完成")
        
        logger.info("加载学生模型...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_path)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        student_kwargs = {
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
            student_kwargs["quantization_config"] = quantization_config
        
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_path,
            **student_kwargs
        )
        logger.info("学生模型加载完成")
        
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        logger.info(f"教师模型参数量: {teacher_params:,}")
        logger.info(f"学生模型参数量: {student_params:,}")
    
    def compute_distillation_loss(self, teacher_logits, student_logits, labels, temperature):
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                   labels.view(-1), ignore_index=-100)
        
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss
    
    def distill_step(self, inputs, device):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None
        labels = inputs["labels"].to(device) if "labels" in inputs else input_ids.clone()
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        loss, soft_loss, hard_loss = self.compute_distillation_loss(
            teacher_logits, student_logits, labels, self.temperature
        )
        
        return loss, soft_loss, hard_loss
    
    def train_distillation(self,
                          train_data_path: str,
                          val_data_path: Optional[str] = None,
                          output_dir: str = "models/distilled",
                          num_epochs: int = 3,
                          batch_size: int = 4,
                          learning_rate: float = 5e-5,
                          gradient_accumulation_steps: int = 4):
        self.load_models()
        
        logger.info(f"加载训练数据: {train_data_path}")
        train_dataset = load_dataset("json", data_files={"train": train_data_path}, split="train")
        
        def tokenize_function(examples):
            texts = examples["text"]
            tokenized = self.student_tokenizer(
                texts,
                truncation=True,
                max_length=2048,
                padding=False
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        
        eval_dataset = None
        if val_data_path:
            eval_dataset = load_dataset("json", data_files={"validation": val_data_path}, split="validation")
            eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.student_tokenizer,
            mlm=False
        )
        
        class DistillationTrainerCustom(Trainer):
            def __init__(self, teacher_model, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.temperature = 5.0
                self.alpha = 0.5
            
            def compute_loss(self, model, inputs, return_outputs=False):
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                labels = inputs.get("labels", input_ids.clone())
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits
                
                student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits
                
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
                
                hard_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                
                return (loss, student_outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to="tensorboard",
        )
        
        trainer = DistillationTrainerCustom(
            teacher_model=self.teacher_model,
            model=self.student_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("开始训练...")
        trainer.train()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model()
        self.student_tokenizer.save_pretrained(output_dir)
        
        logger.info(f"模型已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/distilled")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--use-quantization", action="store_true")
    
    args = parser.parse_args()
    
    trainer = DistillationTrainer(
        teacher_model_path=args.teacher_model,
        student_model_path=args.student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        use_quantization=args.use_quantization
    )
    
    trainer.train_distillation(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()

