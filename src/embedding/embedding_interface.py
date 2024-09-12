import os
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import List
from src.embedding.embedding_configs import EMBEDDING_CONFIGS

# Load environment variables from .env file
load_dotenv()


class UnifiedEmbeddingInterface:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.clients = {}
        self.total_cost = 0

    def get_hf_client(self, model_name):
        if model_name not in self.clients:
            self.clients[model_name] = InferenceClient(model_name, token=self.hf_token)
        return self.clients[model_name]

    def get_openai_client(self):
        if 'openai' not in self.clients:
            self.clients['openai'] = OpenAI(api_key=self.openai_api_key)
        return self.clients['openai']

    def get_openai_embedding(self, model_name, text: str, price_per_1k_tokens: float):
        client = self.get_openai_client()
        response = client.embeddings.create(
            model=model_name,
            input=[text]
        )
        tokens = response.usage.total_tokens
        self.total_cost += (tokens / 1000) * price_per_1k_tokens
        return response.data[0].embedding

    def compute_similarities_hf(self, model_name: str, source_sentence: str, sentences: List[str]):
        client = self.get_hf_client(model_name)
        payload = {
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": sentences
            },
        }
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json=payload
        )
        response_json = response.json()
        return response_json

    def compute_similarities_openai(self, model_name: str, source_sentence: str, sentences: List[str], price: float):
        source_sentence_embedding = self.get_openai_embedding(model_name, source_sentence, price)
        sentences_embeddings = [self.get_openai_embedding(model_name, sentence, price) for sentence in sentences]
        similarities = [np.dot(source_sentence_embedding, embedding) for embedding in sentences_embeddings]
        return similarities

    def compute_similarities(self, model_name: str, source_sentence: str, sentences: List[str]):
        """
         Computes the similarity scores between the source sentence and the list of sentences

         Args:
             model_name (str): The Embedding model name used to compute the similarities.
             source_sentence (str): The source sentence(baseline)
             sentences (int): The candidate sentences to be compared with the source.

         Returns:
             List[float]: The similarity scores between the sentences and the source sentence
         """
        model_info = EMBEDDING_CONFIGS[model_name]
        provider = model_info['provider']
        model_name = model_info['model']
        price = model_info.get('price_per_1k_tokens', 0)
        if provider == 'openai':
            return self.compute_similarities_openai(model_name, source_sentence, sentences, price)
        elif provider == 'huggingface':
            return self.compute_similarities_hf(model_name, source_sentence, sentences)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_total_cost(self):
        return self.total_cost
