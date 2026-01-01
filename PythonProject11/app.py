# -*- coding: utf-8 -*-
import gradio as gr
from model_loader import ModelLoader
from qa_system import QASystem
from config import (
    DEFAULT_MODEL_NAME, USE_QUANTIZATION,
    GRADIO_HOST, GRADIO_PORT
)
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_system = None
model_loader = None


def initialize_system(model_name: str = None, 
                     use_quantization: bool = None):
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    if use_quantization is None:
        use_quantization = USE_QUANTIZATION
    
    global qa_system, model_loader
    
    try:
        logger.info("初始化系统...")
        
        model_loader = ModelLoader(
            model_name=model_name,
            use_quantization=use_quantization
        )
        model_loader.load_model()
        
        qa_system = QASystem(
            model=model_loader.get_model(),
            tokenizer=model_loader.get_tokenizer()
        )
        
        logger.info("系统初始化完成")
        return "系统初始化成功！可以开始提问了。"
        
    except Exception as e:
        error_msg = f"系统初始化失败: {str(e)}"
        logger.error(error_msg)
        return error_msg


def chat(message, history):
    global qa_system
    
    if qa_system is None:
        return history + [[message, "系统尚未初始化，请先在配置中初始化系统。"]]
    
    if not message.strip():
        return history
    
    try:
        answer = qa_system.generate_answer(
            question=message,
            max_length=512,
            temperature=0.7
        )
        
        return history + [[message, answer]]
        
    except Exception as e:
        error_msg = f"处理问题时出错: {str(e)}"
        logger.error(error_msg)
        return history + [[message, error_msg]]


def clear_conversation():
    global qa_system
    if qa_system:
        qa_system.clear_history()
    return []


def create_interface():
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', sans-serif;
    }
    """
    
    with gr.Blocks(css=custom_css, title="本地NLP问答系统") as demo:
        gr.Markdown("""
        # 本地自然语言处理应答系统
        ## 基于Llama2-7B的智能问答系统
        
        这是一个本地化部署的自然语言处理问答系统，支持中文对话。
        """)
        
        with gr.Tab("对话界面"):
            chatbot = gr.Chatbot(
                label="对话窗口",
                height=500,
                show_label=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入您的问题",
                    placeholder="请输入您的问题...",
                    scale=4
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary")
        
        with gr.Tab("系统配置"):
            gr.Markdown("### 模型配置")
            model_name_input = gr.Textbox(
                label="模型名称或路径",
                value=DEFAULT_MODEL_NAME,
                placeholder="输入模型名称或本地路径"
            )
            use_quantization_checkbox = gr.Checkbox(
                label="使用4-bit量化（节省显存）",
                value=USE_QUANTIZATION
            )
            init_btn = gr.Button("初始化系统", variant="primary")
            init_status = gr.Textbox(
                label="初始化状态",
                value="系统尚未初始化",
                interactive=False
            )
        
        submit_btn.click(
            chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "", None, [msg]
        )
        
        msg.submit(
            chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "", None, [msg]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot]
        )
        
        init_btn.click(
            initialize_system,
            inputs=[model_name_input, use_quantization_checkbox],
            outputs=[init_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False
    )

