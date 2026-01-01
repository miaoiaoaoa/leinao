# -*- coding: utf-8 -*-
import torch
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QASystem:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        
    def format_prompt(self, question: str, history: Optional[List] = None) -> str:
        system_prompt = "<s>[INST] <<SYS>>\n你是一个有用的AI助手，请用中文回答问题。\n<</SYS>>\n\n"
        
        if history:
            for item in history[-3:]:
                system_prompt += f"{item['question']} [/INST] {item['answer']} </s><s>[INST] "
        
        system_prompt += f"{question} [/INST]"
        return system_prompt
    
    def generate_answer(self, 
                       question: str, 
                       max_length: int = 512,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       do_sample: bool = True) -> str:
        try:
            prompt = self.format_prompt(question, self.conversation_history)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            self.conversation_history.append({
                'question': question,
                'answer': answer
            })
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            return f"抱歉，处理您的问题时出现了错误: {str(e)}"
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        return self.conversation_history.copy()

