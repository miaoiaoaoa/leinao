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
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_pruning import ModelPruner, load_model_for_pruning
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedTrainer:
    def __init__(self,
                 base_model_path: str,
                 output_dir: str = "models/optimized",
                 pruning_ratio: float = 0.2,
                 use_distillation: bool = False,
                 teacher_model_path: str = None,
                 distillation_temperature: float = 5.0,
                 distillation_alpha: float = 0.5,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 use_quantization: bool = False):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pruning_ratio = pruning_ratio
        self.use_distillation = use_distillation
        self.teacher_model_path = teacher_model_path or base_model_path
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.use_quantization = use_quantization
        
        self.model = None
        self.tokenizer = None
        self.teacher_model = None
    
    def step1_pruning(self):
        logger.info("步骤1: 模型裁剪")
        model, tokenizer = load_model_for_pruning(
            self.base_model_path,
            self.use_quantization
        )
        
        pruner = ModelPruner(model, tokenizer, self.pruning_ratio)
        pruned_model = pruner.prune_model_magnitude()
        
        pruned_model_path = self.output_dir / "pruned_model"
        pruned_model_path.mkdir(exist_ok=True)
        pruned_model.save_pretrained(pruned_model_path)
        tokenizer.save_pretrained(pruned_model_path)
        
        self.base_model_path = str(pruned_model_path)
        self.model = pruned_model
        self.tokenizer = tokenizer
        
        return pruned_model, tokenizer
    
    def step2_load_teacher_for_distillation(self):
        if not self.use_distillation:
            return None
        
        logger.info("步骤2: 加载教师模型")
        teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        
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
        
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"教师模型: {teacher_params:,} 参数")
        logger.info(f"学生模型: {student_params:,} 参数")
        
        return self.teacher_model
    
    def step3_apply_lora(self):
        logger.info("步骤3: 应用LoRA")
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train_integrated(self,
                        train_data_path: str,
                        val_data_path: str = None,
                        num_epochs: int = 3,
                        batch_size: int = 4,
                        learning_rate: float = 2e-4,
                        gradient_accumulation_steps: int = 4):
        logger.info("开始整合训练流程")
        
        if self.pruning_ratio > 0:
            self.step1_pruning()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }
            
            if self.use_quantization:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )
        
        if self.use_distillation:
            self.step2_load_teacher_for_distillation()
        
        self.step3_apply_lora()
        
        logger.info("步骤4: 开始训练")
        train_dataset = load_dataset("json", data_files={"train": train_data_path}, split="train")
        
        def tokenize_function(examples):
            texts = examples["text"]
            tokenized = self.tokenizer(
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
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        class IntegratedTrainerCustom(Trainer):
            def __init__(self, teacher_model, use_distillation, temperature, alpha, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_model = teacher_model
                self.use_distillation = use_distillation
                self.temperature = temperature
                self.alpha = alpha
            
            def compute_loss(self, model, inputs, return_outputs=False):
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                labels = inputs.get("labels", input_ids.clone())
                
                student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits
                
                if self.use_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        teacher_logits = teacher_outputs.logits
                    
                    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                    student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
                    
                    hard_loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    
                    loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                else:
                    loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                return (loss, student_outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
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
        
        trainer = IntegratedTrainerCustom(
            teacher_model=self.teacher_model,
            use_distillation=self.use_distillation,
            temperature=self.distillation_temperature,
            alpha=self.distillation_alpha,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("开始训练...")
        trainer.train()
        
        final_model_path = self.output_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model()
        self.tokenizer.save_pretrained(final_model_path)
        
        lora_path = self.output_dir / "lora_adapter"
        lora_path.mkdir(exist_ok=True)
        self.model.save_pretrained(str(lora_path))
        
        logger.info("训练完成")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/optimized")
    parser.add_argument("--pruning-ratio", type=float, default=0.2)
    parser.add_argument("--use-distillation", action="store_true")
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--distillation-temperature", type=float, default=5.0)
    parser.add_argument("--distillation-alpha", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--use-quantization", action="store_true")
    
    args = parser.parse_args()
    
    trainer = IntegratedTrainer(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        pruning_ratio=args.pruning_ratio,
        use_distillation=args.use_distillation,
        teacher_model_path=args.teacher_model,
        distillation_temperature=args.distillation_temperature,
        distillation_alpha=args.distillation_alpha,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_quantization=args.use_quantization
    )
    
    trainer.train_integrated(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()

