import os
import requests
import uuid
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MistralChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"  # Updated endpoint for chat completions
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.knowledge_base = []

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF file."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text

    def load_knowledge_base(self, pdf_paths, chunk_size=500, overlap=100):
        """Loads and embeds text chunks from PDFs."""
        for pdf_path in pdf_paths:
            text = self.extract_text_from_pdf(pdf_path)
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            embeddings = self.model.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.knowledge_base.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "embedding": embedding,
                    "source": pdf_path,
                    "chunk_index": i
                })
        
        print(f"Knowledge base loaded with {len(self.knowledge_base)} chunks from PDFs.")

    def retrieve_context(self, query, top_k=3):
        """Retrieves the most relevant context for a query."""
        query_embedding = self.model.encode([query])[0]
        scores = [
            (item, cosine_similarity([query_embedding], [item["embedding"]])[0][0])
            for item in self.knowledge_base
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0]["text"] for item in scores[:top_k]]

    def generate_response(self, query):
        """Generates a response from the Mistral API based on relevant context."""
        contexts = self.retrieve_context(query)
        context_text = "\n".join(contexts)

        payload = {
            "model": "mistral-tiny",  # Updated model name
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that uses the provided context to answer questions."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return "Sorry, there was an error processing your request."

chatbot = MistralChatbot(api_key="API KEY")