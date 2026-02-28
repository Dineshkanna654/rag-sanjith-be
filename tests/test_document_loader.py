import pytest
from pathlib import Path
from app.services.document_loader import load_and_split_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_txt_file():
    chunks = load_and_split_file(FIXTURES / "sample.txt", "sample.txt")
    assert len(chunks) >= 1
    assert any("test document" in chunk.page_content for chunk in chunks)


def test_unsupported_file_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_and_split_file(Path("/tmp/fake.csv"), "fake.csv")
