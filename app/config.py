from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    DATABASE_URL: str = "postgresql+asyncpg://sanjithkrishnab@localhost:5432/enterprise_rag"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
