EMBEDDING_CONFIGS = {
    "openai_embedding_3_small": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "openai_embedding_3_large": {
        "provider": "openai",
        "model": "text-embedding-3-large"
    },
    "openai_embedding_2_ada": {
        "provider": "openai",
        "model": "text-embedding-ada-002"
    },
    "bge_m3": {
        "provider": "huggingface",
        "model": "BAAI/bge-m3"
    },
    "all_MiniLM_L12_v2": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L12-v2 "
    },
    "all_MiniLM_L6_v2": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2 "
    },
    "gte_Qwen2": {
        "provider": "huggingface",
        "model": "Alibaba-NLP/gte-Qwen2-7B-instruct"
    }
}
