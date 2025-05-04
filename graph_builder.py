# graph_builder.py
"""
Functions for building and saving the knowledge graph using NetworkX.
"""

import networkx as nx
from typing import List, Tuple, Callable
from llm_interaction import AutoModelForCausalLM, AutoTokenizer # For type hinting
import logging
from tqdm import tqdm # For progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type hint for the extraction function
ExtractionFunction = Callable[[str, AutoModelForCausalLM, AutoTokenizer, str, int, float, float, bool], List[Tuple[str, str, str]]]

def build_knowledge_graph(texts: List[str],
                          model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer,
                          extract_knowledge_func: ExtractionFunction,
                          prompt_template: str,
                          max_new_tokens: int,
                          temperature: float,
                          top_p: float,
                          debug_limit: int = 0) -> nx.DiGraph:
    """
    Builds a directed knowledge graph from a list of texts using an extraction function.

    Args:
        texts (List[str]): A list of text documents to process.
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        extract_knowledge_func (ExtractionFunction): The function to use for extracting triples.
        prompt_template (str): The template for the extraction prompt.
        max_new_tokens (int): Max tokens for generation in extraction.
        temperature (float): Temperature for generation in extraction.
        top_p (float): Top_p for generation in extraction.
        debug_limit (int): Number of initial texts to process with debug=True.

    Returns:
        nx.DiGraph: The constructed knowledge graph.
    """
    knowledge_graph = nx.DiGraph()
    logging.info(f"Starting knowledge graph construction from {len(texts)} texts.")

    processed_texts = 0
    total_triples = 0

    # Wrap texts with tqdm for a progress bar
    for i, text in enumerate(tqdm(texts, desc="Processing Texts")):
        is_debug = (i < debug_limit)
        try:
            triples = extract_knowledge_func(
                text=text,
                model=model,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                debug=is_debug
            )
            processed_texts += 1
            if triples:
                for entity1, relation, entity2 in triples:
                    # Add nodes (NetworkX handles duplicates automatically)
                    knowledge_graph.add_node(entity1, label=entity1)
                    knowledge_graph.add_node(entity2, label=entity2)
                    # Add edge with relation as an attribute
                    knowledge_graph.add_edge(entity1, entity2, relation=relation)
                    total_triples += 1
        except Exception as e:
            logging.error(f"Failed to process text index {i}: {e}", exc_info=True)
            # Decide whether to continue or stop on error
            # continue

    logging.info(f"Finished processing {processed_texts}/{len(texts)} texts.")
    logging.info(f"Built knowledge graph with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges ({total_triples} triples added).")
    return knowledge_graph


def save_graph(graph: nx.DiGraph, filepath: str):
    """
    Saves the NetworkX graph to a file (GraphML format).

    Args:
        graph (nx.DiGraph): The graph to save.
        filepath (str): The path where the graph file will be saved.
    """
    try:
        nx.write_graphml(graph, filepath)
        logging.info(f"Knowledge graph successfully saved to '{filepath}'")
    except Exception as e:
        logging.error(f"Failed to save graph to '{filepath}': {e}")