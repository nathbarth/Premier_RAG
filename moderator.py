"""
Brique 2 - L'agent moderateur.
"""

import json

from groq import Groq

import config


class Moderator:
    def __init__(self, client: Groq):
        self.client = client
        with open(config.MODERATOR_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question: str) -> dict:
        response = self.client.chat.completions.create(
            model=config.MODERATION_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
        )

        raw_content = response.choices[0].message.content

        try:
            decision = json.loads(raw_content)
        except json.JSONDecodeError:
            return {"is_prompt_injection": True, "raw_response": raw_content}

        return decision