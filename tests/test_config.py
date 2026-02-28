from app.config import settings


def test_settings_have_defaults():
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.MODEL_NAME == "llama3.2"
    assert settings.EMBEDDING_MODEL == "nomic-embed-text"
    assert settings.CHROMA_PERSIST_DIR == "./chroma_db"
