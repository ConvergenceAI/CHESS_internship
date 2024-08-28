PIPELINE_CONFIG = {
    "keyword_extraction": {
        "model": "gpt-3.5",
    },
    "entity_retrieval": {
        "embedding_model_name": "openai_embedding_3_large",
    },
    "context_retrieval": {
        "top_k": 5,
    },
    "column_filtering": {
        "model": "llama-3",
    },
    "table_selection": {
        "model": "gpt-4",
        "num_samples": 1,
    },
    "column_selection": {
        "model": "gpt-4",
        "num_samples": 1,
    },
    "candidate_generation": {
        "model": "gpt-4",
        "num_samples": 1,
    },
    "revision": {
        "model": "gpt-4",
        "num_samples": 1,
    }
}
