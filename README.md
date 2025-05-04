# Knowledge_Graph_with_LLM# Perfected-KG: LLM-Powered Knowledge Graph Builder

This project uses a Large Language Model (LLM) to extract knowledge triples (Entity1, Relation, Entity2) from text data and build a knowledge graph using NetworkX.

## Project Structure

- `config.py`: Contains configuration settings like model names, dataset details, file paths, and LLM parameters.
- `llm_interaction.py`: Handles loading the LLM and tokenizer, and the core function to extract knowledge triples from text using prompts.
- `graph_builder.py`: Contains functions to construct the NetworkX graph from the extracted triples and save it to a file.
- `main.py`: The main script that orchestrates the workflow: loads data, loads the model, extracts knowledge, builds the graph, and saves the result.
- `requirements.txt`: Lists the necessary Python libraries.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd perfected-kg
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Ensure you have Python 3.8+ and pip installed. Depending on your system and GPU, you might need specific versions of `torch`, `bitsandbytes`, etc. Install PyTorch first following instructions on the [official PyTorch website](https://pytorch.org/).
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `bitsandbytes` installation might require specific system libraries.*

## Configuration

Adjust settings in `config.py` as needed:
- `DATASET_NAME`, `DATASET_SPLIT`, `TEXT_COLUMN`: Specify the Hugging Face dataset and the column containing text.
- `MODEL_NAME`: Choose the LLM (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`).
- `QUANTIZATION_CONFIG`: Modify quantization settings if desired (or set to `None` to disable).
- `OUTPUT_GRAPH_FILE`: Change the output filename for the graph.
- Extraction parameters (`MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`).

## Running the Pipeline

Execute the main script:

```bash
python main.py



The primary output is a GraphML file (e.g., dermatology_knowledge_graph.graphml) representing the extracted knowledge graph. This file can be loaded and analyzed using NetworkX or other graph analysis tools.