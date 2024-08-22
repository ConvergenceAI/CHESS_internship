import os
from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from src.llm.model_configs import MODEL_CONFIGS
# from transformers import pipeline

# Load environment variables from .env file
load_dotenv()


class UnifiedLLMInterface:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.clients = {}

    def get_hf_client(self, model_name):
        if model_name not in self.clients:
            self.clients[model_name] = InferenceClient(model_name, token=self.hf_token)
        return self.clients[model_name]

    def get_openai_client(self):
        if 'openai' not in self.clients:
            self.clients['openai'] = OpenAI(api_key=self.openai_api_key)
        return self.clients['openai']

    def generate_openai_chat(self, model_name, prompt, temperature, max_tokens):
        client = self.get_openai_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def generate_openai_completion(self, model_name, prompt, temperature, max_tokens):
        client = self.get_openai_client()
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()

    def generate_huggingface_inference(self, model_name, prompt, temperature, max_tokens):
        client = self.get_hf_client(model_name)
        response = ""
        for message in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
        ):
            response += message.choices[0].delta.content
        return response

    # def generate_huggingface_transformers(self, model_name, prompt):
    #     messages = [
    #         {"role": "user", "content": prompt}, ]
    #     pipe = pipeline("text-generation", model=model_name,framework='pt')
    #     pipe(messages)

    def generate(self, model_name, prompt):
        model_info = MODEL_CONFIGS[model_name]
        provider = model_info['provider']
        model_name = model_info['model']
        temperature = model_info.get('temperature', 0.7)
        max_tokens = model_info.get('max_tokens')
        model_type = model_info.get('model_type', '')
        if provider == 'openai':
            if model_type == 'chat':
                return self.generate_openai_chat(model_name, prompt, temperature, max_tokens)
            else:
                return self.generate_openai_completion(model_name, prompt, temperature, max_tokens)
        elif provider == 'huggingface':
            if model_type == 'inference':
                return self.generate_huggingface_inference(model_name, prompt, temperature, max_tokens)
            # else:
            #     return self.generate_huggingface_transformers(model_name, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
