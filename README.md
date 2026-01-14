Hybrid RAG Question Answering System

A Hybrid Retrieval-Augmented Generation (RAG) system that combines real-time web search with offline local knowledge retrieval to generate accurate, low-hallucination answers using open-source LLMs. The system is optimized for cost efficiency, privacy, and fast inference.

🚀 Features

Hybrid RAG Architecture
Grounds LLM responses using retrieved context via FAISS, significantly reducing hallucinations.

Dual Retrieval Modes

Online Mode: Fetches real-time information using Serper API.

Offline Mode: Performs fast semantic search over local data using SBERT embeddings.

Incremental Knowledge Caching
Automatically stores online search results into a local FAISS index, enabling zero-cost, private inference for repeated queries.

Low Latency
Achieves sub-3 second response time for offline document retrieval.

Interactive Streamlit UI
Professional chat interface with conversation history, source citations, and animated components.
