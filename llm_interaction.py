# llm_interaction.py
"""
Functions for interacting with the Language Model (LLM)
for knowledge extraction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple, Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_name: str,
                             quantization_config: Optional[BitsAndBytesConfig] = None,
                             device_map: str = "auto",
                             trust_remote_code: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the LLM and its tokenizer with optional quantization.

    Args:
        model_name (str): The name of the model from Hugging Face Hub.
        quantization_config (Optional[BitsAndBytesConfig]): Configuration for quantization.
        device_map (str): Device mapping strategy (e.g., "auto", "cuda:0").
        trust_remote_code (bool): Whether to trust remote code for model loading.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

    Raises:
        ValueError: If model loading fails.
    """
    try:
        logging.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )
        # Set pad token ID to EOS token ID if not already set for open-end generation
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
             model.config.pad_token_id = model.config.eos_token_id
             logging.warning(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")

        logging.info(f"Successfully loaded {model_name}" + (" with 4-bit quantization." if quantization_config and quantization_config.load_in_4bit else "."))
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise ValueError(f"Failed to load model or tokenizer: {e}")


def extract_knowledge(text: str,
                      model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer,
                      prompt_template: str,
                      max_new_tokens: int = 200,
                      temperature: float = 0.6,
                      top_p: float = 0.95,
                      debug: bool = False) -> List[Tuple[str, str, str]]:
    """
    Extracts knowledge triples (Entity1, Relation, Entity2) from text using the LLM.

    Args:
        text (str): The input text to extract knowledge from.
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        prompt_template (str): The template for the extraction prompt (must contain '{text}').
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generation.
        top_p (float): Nucleus sampling probability.
        debug (bool): If True, print the input text and model response.

    Returns:
        List[Tuple[str, str, str]]: A list of extracted (entity1, relation, entity2) triples.
    """
    if not text or not text.strip():
        logging.warning("Received empty text for extraction, returning empty list.")
        return []

    prompt = prompt_template.format(text=text)
    triples = []

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Ensure pad_token_id is set for generation
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": tokenizer.eos_token_id # Use EOS token for padding in generation
        }

        outputs = model.generate(**inputs, **generation_kwargs)
        # Decode, skipping special tokens and the prompt part
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        if debug:
            logging.info(f"\n--- Debug Extraction ---")
            logging.info(f"Input Text: {text[:500]}...") # Limit printing long texts
            logging.info(f"Generated Response Fragment: {response}")
            logging.info(f"------------------------")

        # Robust parsing with regex to find "Entity, Relation, Entity" patterns
        # This regex looks for:
        #   - Group 1: One or more characters (non-greedy) before the first comma
        #   - A comma, possibly surrounded by whitespace
        #   - Group 2: One or more characters (non-greedy) before the second comma
        #   - A comma, possibly surrounded by whitespace
        #   - Group 3: One or more characters until the end of the line
        for line in response.split("\n"):
            line = line.strip()
            # Simple pattern: expects exactly three parts separated by commas
            match = re.match(r"^([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+)$", line)
            if match:
                entity1, relation, entity2 = match.groups()
                # Normalize: lowercase and strip extra whitespace
                entity1 = entity1.strip().lower()
                entity2 = entity2.strip().lower()
                relation = relation.strip().lower()
                # Basic validation: ensure parts are not empty
                if entity1 and relation and entity2:
                    triples.append((entity1, relation, entity2))
                    if debug:
                        logging.info(f"  Extracted Triple: {entity1}, {relation}, {entity2}")
            elif debug and line: # Log lines that didn't match if debugging and not empty
                logging.info(f"  Skipped Line (no match): {line}")

    except Exception as e:
        logging.error(f"Error during knowledge extraction for text starting with '{text[:50]}...': {e}", exc_info=True)
        # Optionally return partial results or raise the error
        # return triples # Return whatever was extracted before the error

    return triples