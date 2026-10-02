from google import genai
import os
import time


class RAG:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)

    def generate_answer(self, query, retrieved_chunks):

        context = "\n\n".join(
            f"Source {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )

        prompt = f"""
You are BioNexus, an evidence-based medical research assistant.

Answer the user's question using ONLY the research evidence below.
If the evidence is insufficient, say so clearly.

Question:
{query}

Research evidence:
{context}

Give a concise evidence-based answer.
"""

        # Try Gemini 3.5 Flash-Lite
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                return response.text

            except Exception as e:
                if attempt == 2:
                    raise e

                time.sleep(2)