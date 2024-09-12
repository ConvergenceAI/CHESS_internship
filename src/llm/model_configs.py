MODEL_CONFIGS = {
    "gpt-3.5": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 4000,
        "model_type": "chat",
        "price_per_1k_input_tokens": 0.0005,
        "price_per_1k_output_tokens": 0.0015

    },
    "gpt-4": {
        "provider": "openai",
        "model": "gpt-4-turbo",
        "temperature": 0,
        "model_type": "chat",
        "max_tokens": 4000,
        "price_per_1k_input_tokens": 0.01,
        "price_per_1k_output_tokens": 0.03
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0,
        "model_type": "chat",
        "max_tokens": 4000,
        "price_per_1k_input_tokens": 0.005,
        "price_per_1k_output_tokens": 0.015
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "model_type": "chat",
        "max_tokens": 4000,
        "price_per_1k_input_tokens": 0.00015,
        "price_per_1k_output_tokens": 0.0006
    },
    "mixtral": {
        "provider": "huggingface",
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0,
        "model_type": "inference",
        "max_tokens": 4000
    },
    "llama-3": {
        "provider": "huggingface",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0,
        "model_type": "inference",
        "max_tokens": 4000
    },
    "phi-3": {
        "provider": "huggingface",
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "temperature": 0,
        "model_type": "inference",
        "max_tokens": 4000
    },
    "Athene": {
        "provider": "huggingface",
        "model": "Nexusflow/Athene-70B",
        "temperature": 0,
        "model_type": "inference",
        "max_tokens": 4000
    },
    "Yi-1.5": {
        "provider": "huggingface",
        "model": "01-ai/Yi-1.5-34B-Chat",
        "temperature": 0,
        "model_type": "inference",
        "max_tokens": 4000
    },
    "finetuned-NL2SQL": {
        "provider": "huggingface",
        "model": "AI4DS/NL2SQL_DeepSeek_33B",
        "temperature": 0,
        "model_type": "transformers",
        "max_tokens": 4000

    }
}
