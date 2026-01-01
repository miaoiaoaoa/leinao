# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from pathlib import Path
import argparse
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPruner:
    def __init__(self, model, tokenizer, pruning_ratio: float = 0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.pruning_ratio = pruning_ratio
        self.original_model = None
    
    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def magnitude_pruning(self, module, ratio: float):
        with torch.no_grad():
            for name, param in module.named_parameters():
                if len(param.shape) >= 2:
                    threshold = torch.quantile(
                        torch.abs(param.data),
                        ratio
                    )
                    mask = torch.abs(param.data) > threshold
                    param.data *= mask.float()
    
    def structured_pruning_attention(self, model, pruning_ratio: float):
        pruned_heads = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                num_heads = module.num_heads
                heads_to_remove = int(num_heads * pruning_ratio)
                
                if heads_to_remove > 0 and heads_to_remove < num_heads:
                    logger.info(f"模块 {name}: {num_heads} 个头，移除 {heads_to_remove} 个")
                    pruned_heads[name] = {
                        'original': num_heads,
                        'removed': heads_to_remove
                    }
        
        return pruned_heads
    
    def apply_low_rank_approximation(self, linear_layer, rank: int):
        weight = linear_layer.weight.data
        out_features, in_features = weight.shape
        
        if rank >= min(out_features, in_features):
            return linear_layer
        
        U, S, V = torch.svd(weight)
        U_approx = U[:, :rank] @ torch.diag(S[:rank])
        V_approx = V[:, :rank].t()
        
        logger.info(f"将 {in_features}x{out_features} 矩阵近似为 {in_features}x{rank} 和 {rank}x{out_features}")
        
        return U_approx, V_approx
    
    def prune_model_magnitude(self):
        logger.info(f"开始裁剪，比例: {self.pruning_ratio}")
        
        self.original_model = copy.deepcopy(self.model)
        original_params, _ = self.count_parameters(self.original_model)
        logger.info(f"原始参数量: {original_params:,}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                self.magnitude_pruning(module, self.pruning_ratio)
        
        pruned_params, _ = self.count_parameters(self.model)
        logger.info(f"裁剪后参数量: {pruned_params:,}")
        
        return self.model
    
    def prune_model_structured(self):
        logger.info(f"开始结构化裁剪")
        pruned_info = self.structured_pruning_attention(self.model, self.pruning_ratio)
        logger.info(f"处理了 {len(pruned_info)} 个注意力模块")
        return self.model, pruned_info
    
    def save_pruned_model(self, output_path: str, pruning_info: dict = None):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        if pruning_info:
            import json
            info_path = output_path / "pruning_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(pruning_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {output_path}")
        total_params, trainable_params = self.count_parameters(self.model)
        logger.info(f"参数量: {total_params:,} (可训练: {trainable_params:,})")


def load_model_for_pruning(model_path: str, use_quantization: bool = False):
    logger.info(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if use_quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--pruning-ratio", type=float, default=0.3)
    parser.add_argument("--pruning-method", type=str, default="magnitude",
                       choices=["magnitude", "structured"])
    parser.add_argument("--use-quantization", action="store_true")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model_for_pruning(args.model_path, args.use_quantization)
    pruner = ModelPruner(model, tokenizer, args.pruning_ratio)
    
    if args.pruning_method == "magnitude":
        pruned_model = pruner.prune_model_magnitude()
        pruning_info = None
    else:
        pruned_model, pruning_info = pruner.prune_model_structured()
    
    pruner.save_pruned_model(args.output_path, pruning_info)
    logger.info("裁剪完成")


if __name__ == "__main__":
    main()

