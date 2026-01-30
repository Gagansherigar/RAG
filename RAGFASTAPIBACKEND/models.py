from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RAGType(str, Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"
    PIPELINE = "pipeline"


class QueryRequest(BaseModel):
    query: str
    rag_type: RAGType = RAGType.SIMPLE
    top_k: int = 3
    min_score: float = 0.2
    return_context: bool = False
    summarize: bool = False


class SourceInfo(BaseModel):
    source: str
    page: str
    score: float
    preview: str


class SimpleResponse(BaseModel):
    answer: str
    query: str
    rag_type: str


class AdvancedResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: float
    context: Optional[str] = None
    query: str
    rag_type: str

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is...",
                "sources": [{"source": "doc.pdf", "page": "1", "score": 0.85, "preview": "ML is..."}],
                "confidence": 0.85,
                "context": None,
                "query": "What is ML?",
                "rag_type": "advanced"
            }
        }


class HistoryItem(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    summary: Optional[str] = None


class PipelineResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    summary: Optional[str] = None
    history: List[HistoryItem]
    rag_type: str