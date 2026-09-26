# app/db/models.py
from dataclasses import dataclass


@dataclass
class Document:
    id: int
    filename: str


@dataclass
class Chunk:
    id: int
    document_id: int
    text: str
    embedding: list[float]