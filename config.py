# config.py
"""
Configuration settings for the Knowledge Graph Builder.
"""

import torch
from transformers import BitsAndBytesConfig

# --- Dataset Configuration ---
DATASET_NAME = "Carxofa85/dermatology"
DATASET_SPLIT = "train"
TEXT_COLUMN = "output" # The column containing the text data

# --- LLM Configuration ---
# Choose model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
TRUST_REMOTE_CODE = True

# --- Quantization Configuration ---
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
DEVICE_MAP = "auto" # Or specify GPU e.g., "cuda:0"

# --- Knowledge Extraction Parameters ---
EXTRACTION_PROMPT_TEMPLATE = """
Given the following text, extract entities (nodes) and their relations in the format:
Entity1, Relation, Entity2
Separate each triple with a newline. If no clear relations are found, return an empty list.
Text: {text}
"""
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.6
TOP_P = 0.95
DEBUG_EXTRACTION_LIMIT = 3 # Print detailed extraction info for the first N texts

# --- Knowledge Graph Configuration ---
OUTPUT_GRAPH_FILE = "dermatology_knowledge_graph.graphml"