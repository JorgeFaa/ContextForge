from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


settings = Settings()
