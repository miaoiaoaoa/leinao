# -*- coding: utf-8 -*-
import os

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"

GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "")

