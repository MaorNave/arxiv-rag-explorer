"""arXiv RAG Explorer: local retrieval-augmented generation over research abstracts."""

import os

# Keep the pipeline fully local: no telemetry from the vector database.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

__version__ = "1.0.0"
