# tests/test_loader.py
from pathlib import Path

import pytest

from app.ingestion.loader import load_document


def test_load_document_reads_txt_file(tmp_path: Path):
    file_path = tmp_path / "prueba.txt"
    file_path.write_text("Hola mundo", encoding="utf-8")

    result = load_document(file_path)

    assert result == "Hola mundo"


def test_load_document_reads_md_file(tmp_path: Path):
    file_path = tmp_path / "prueba.md"
    file_path.write_text("# Titulo\nContenido", encoding="utf-8")

    result = load_document(file_path)

    assert result == "# Titulo\nContenido"


def test_load_document_raises_for_unsupported_extension(tmp_path: Path):
    file_path = tmp_path / "prueba.docx"
    file_path.write_text("contenido", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)
