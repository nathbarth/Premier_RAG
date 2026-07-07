"""
Brique 3 - Le RAG qui orchestre tout.
"""

import os

from dotenv import load_dotenv
from groq import Groq

import config
from vector_db import VectorDB
from moderator import Moderator


class RAG:
    def __init__(self, vector_db: VectorDB, use_moderator: bool = True):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY introuvable. Verifiez votre fichier .env."
            )

        self.client = Groq(api_key=api_key)
        self.vector_db = vector_db

        self.use_moderator = use_moderator
        self.moderator = Moderator(self.client) if use_moderator else None

        with open(config.RAG_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, chunks):
        formatted_chunks = "\n".join(f"- {c['text']}" for c in chunks)
        return self.system_prompt_template.replace(
            config.CHUNKS_PLACEHOLDER, formatted_chunks
        )

    def answer_question(self, question: str) -> str:
        if self.use_moderator:
            decision = self.moderator.moderate(question)
            if decision.get("is_prompt_injection"):
                return (
                    "Je ne peux pas traiter cette demande : elle ressemble a une "
                    "tentative de contournement de mes instructions (prompt injection)."
                )

        chunks = self.vector_db.retrieve(question, n=config.N_CHUNKS_RETRIEVED)
        system_prompt = self._build_system_prompt(chunks)

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        return response.choices[0].message.content