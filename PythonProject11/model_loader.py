# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", 
                 use_quantization=True, device_map="auto"):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        try:
            logger.info(f"开始加载模型: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_quantization else None,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            logger.info("模型加载完成")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_model(self):
        if self.model is None:
            raise ValueError("模型尚未加载，请先调用load_model()")
        return self.model
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("分词器尚未加载，请先调用load_model()")
        return self.tokenizer

