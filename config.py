"""
Module de configuration centralise.
"""

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
MODERATION_MODEL_NAME = "meta-llama/llama-guard-4-12b"

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_demo_collection"

N_CHUNKS_RETRIEVED = 3

RAG_SYSTEM_PROMPT_PATH = "prompts/rag_system_prompt.txt"
MODERATOR_SYSTEM_PROMPT_PATH = "prompts/moderator_system_prompt.txt"

CHUNKS_PLACEHOLDER = "{{Chunks}}"