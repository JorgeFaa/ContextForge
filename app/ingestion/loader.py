#app/ingestion/loader.py
from pathlib import Path
from pypdf import PdfReader

def load_document(path: Path) -> str:
    """ lee un archivo y devuelve su texto plano, segun su extension """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Tipo de archivo no soportado: {suffix}")

def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)