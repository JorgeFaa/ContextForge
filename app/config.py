from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "ollama" # "ollama" o "anthropic"
    anthropic_api_key: str | None = None
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"


settings = Settings()
