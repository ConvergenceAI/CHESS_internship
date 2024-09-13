PIPELINE_CONFIG = {
    "keyword_extraction": {
        "model": "gpt-4o-mini",
    },
    "entity_retrieval": {
        "embedding_model_name": "openai_embedding_3_large",
    },
    "context_retrieval": {
        "top_k": 5,
    },
    "column_filtering": {
        "model": "gpt-4o",
    },
    "table_selection": {
        "model": "gpt-4o",
        "num_samples": 1,
    },
    "column_selection": {
        "model": "gpt-4o",
        "num_samples": 1,
    },
    "schema_selection_fusion": {
        "model": "gpt-4o",
        "num_samples": 1,
    },
    "candidate_generation": {
        "model": "gpt-4o",
        "num_samples": 1,
    },
    "revision": {
        "model": "gpt-4o",
        "num_samples": 1,
    }
}
