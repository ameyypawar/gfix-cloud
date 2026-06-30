from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/gfixcloud"

    # LLM generation
    anthropic_api_key: str = ""
    generation_model: str = "claude-haiku-4-5"
    llm_timeout_secs: int = 30

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # gfix binary
    gfix_bin: str = "gfix"
    gfix_version: str = "0.1.0-alpha.6"

    # gfix env flags
    gitfix_allow_any_repo: int = 1
    gitfix_byok: int = 0

    # Retrieval
    hnsw_ef_search: int = 40
    rag_top_k: int = 3

    # Server
    cors_origins: str = "http://localhost:3000"
    port: int = 8080


settings = Settings()
