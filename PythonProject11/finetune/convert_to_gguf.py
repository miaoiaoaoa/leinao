# -*- coding: utf-8 -*-
"""
模型转换脚本
将微调后的模型转换为GGUF格式供Ollama使用
注意：此脚本需要llama.cpp工具，主要用于说明转换流程
"""
import subprocess
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
    """模型转换器（转换为GGUF格式）"""
    
    def __init__(self, model_path: str, output_path: str = None):
        """
        初始化转换器
        
        Args:
            model_path: 模型路径
            output_path: 输出路径
        """
        self.model_path = Path(model_path)
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.model_path.parent / f"{self.model_path.name}.gguf"
    
    def check_llama_cpp(self) -> bool:
        """检查llama.cpp是否可用"""
        try:
            result = subprocess.run(
                ["python", "-c", "import llama_cpp"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def convert_hf_to_gguf(self, quantization: str = "Q4_K_M"):
        """
        将HuggingFace格式转换为GGUF格式
        
        Args:
            quantization: 量化类型 (Q4_K_M, Q5_K_M, Q8_0等)
        
        注意：实际转换需要使用llama.cpp的convert_hf_to_gguf.py脚本
        """
        logger.info(f"开始转换模型: {self.model_path} -> {self.output_path}")
        logger.info(f"量化类型: {quantization}")
        
        # 这里提供转换命令示例，实际需要安装llama.cpp
        convert_script = "convert_hf_to_gguf.py"  # llama.cpp的转换脚本
        
        cmd = [
            "python", convert_script,
            str(self.model_path),
            "--outdir", str(self.output_path.parent),
            "--outtype", quantization
        ]
        
        logger.info(f"转换命令: {' '.join(cmd)}")
        logger.warning("注意：实际转换需要安装llama.cpp工具")
        logger.warning("安装方法: git clone https://github.com/ggerganov/llama.cpp")
        
        # 如果llama.cpp可用，执行转换
        # try:
        #     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        #     logger.info("转换成功！")
        #     logger.info(result.stdout)
        # except subprocess.CalledProcessError as e:
        #     logger.error(f"转换失败: {e.stderr}")
        #     raise
        # except FileNotFoundError:
        #     logger.error("未找到llama.cpp转换脚本，请先安装llama.cpp")
        #     raise
    
    def create_ollama_modelfile(self, model_name: str = "llama2-finetuned",
                                template: str = None, system_prompt: str = None):
        """
        创建Ollama Modelfile
        
        Args:
            model_name: 模型名称
            template: 模板字符串
            system_prompt: 系统提示词
        """
        if template is None:
            template = """{{ if .System }}<s>[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ end }}{{ if .Prompt }}<s>[INST] {{ .Prompt }} [/INST]{{ end }}"""
        
        if system_prompt is None:
            system_prompt = "你是一个有用的AI助手，请用中文回答问题。"
        
        modelfile_content = f"""FROM {self.output_path}

TEMPLATE \"\"\"{template}\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
        
        modelfile_path = self.output_path.parent / "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile已创建: {modelfile_path}")
        logger.info(f"使用以下命令导入到Ollama:")
        logger.info(f"  ollama create {model_name} -f {modelfile_path}")
        
        return modelfile_path


def main():
    parser = argparse.ArgumentParser(description="转换模型为GGUF格式")
    parser.add_argument("--model-path", type=str, required=True,
                       help="模型路径（HuggingFace格式）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出路径")
    parser.add_argument("--quantization", type=str, default="Q4_K_M",
                       choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", 
                               "Q4_K_M", "Q5_K_M"],
                       help="量化类型")
    parser.add_argument("--create-modelfile", action="store_true",
                       help="创建Ollama Modelfile")
    parser.add_argument("--model-name", type=str, default="llama2-finetuned",
                       help="Ollama模型名称")
    
    args = parser.parse_args()
    
    converter = ModelConverter(args.model_path, args.output)
    
    # 检查llama.cpp
    if not converter.check_llama_cpp():
        logger.warning("llama.cpp未安装，无法执行实际转换")
        logger.warning("此脚本主要用于展示转换流程")
    
    # 执行转换（示例）
    converter.convert_hf_to_gguf(args.quantization)
    
    # 创建Modelfile
    if args.create_modelfile:
        converter.create_ollama_modelfile(args.model_name)


if __name__ == "__main__":
    main()

