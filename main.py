"""Start the arXiv RAG Explorer web app on http://127.0.0.1:8080.

    python main.py                                  # uses data/arxiv_2.9k.jsonl
    DATA_PATH=/path/to/other.jsonl python main.py   # any .jsonl dataset
    python main.py --help                           # all CLI options

On first start the app pulls the Ollama models it needs (Qwen 3.5 + an embedding
model), builds the local index and then serves the UI; later starts reuse the index.
"""

from rag_app.__main__ import main

if __name__ == "__main__":
    main()
