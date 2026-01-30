from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
import logging

from models import (
    QueryRequest, SimpleResponse, AdvancedResponse,
    PipelineResponse, RAGType
)
from rag_core import initialize_rag, rag_simple, rag_advanced

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
retriever = None
llm = None
pipeline = None
embedding_manager = None
vector_store = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG on startup"""
    global retriever, llm, pipeline, embedding_manager, vector_store

    logger.info("Initializing RAG system...")
    try:
        retriever, llm, pipeline, embedding_manager, vector_store = initialize_rag()
        logger.info("RAG system ready!")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents": vector_store.get_count() if vector_store else 0,
        "embedding_dim": embedding_manager.get_embeddings_dimension() if embedding_manager else 0
    }


@app.post("/query", response_model=Union[SimpleResponse, AdvancedResponse, PipelineResponse])
async def query(request: QueryRequest):
    """Main query endpoint"""
    try:
        logger.info(f"Received query: {request.query}, type: {request.rag_type}")

        if request.rag_type == RAGType.SIMPLE:
            answer = rag_simple(request.query, retriever, llm, request.top_k)
            return SimpleResponse(
                answer=answer,
                query=request.query,
                rag_type="simple"
            )

        elif request.rag_type == RAGType.ADVANCED:
            result = rag_advanced(
                request.query, retriever, llm,
                request.top_k, request.min_score, request.return_context
            )

            logger.info(f"Advanced RAG result - sources: {len(result['sources'])}, confidence: {result['confidence']}")

            return AdvancedResponse(
                answer=result['answer'],
                sources=result['sources'],
                confidence=result['confidence'],
                context=result.get('context'),
                query=request.query,
                rag_type="advanced"
            )

        elif request.rag_type == RAGType.PIPELINE:
            result = pipeline.query(
                request.query, request.top_k,
                request.min_score, summarize=request.summarize
            )
            return PipelineResponse(
                question=result['question'],
                answer=result['answer'],
                sources=result['sources'],
                summary=result['summary'],
                history=result['history'],
                rag_type="pipeline"
            )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """Get query history"""
    return {"history": pipeline.history if pipeline else []}


@app.delete("/history")
async def clear_history():
    """Clear query history"""
    if pipeline:
        pipeline.clear_history()
    return {"message": "History cleared"}


@app.post("/process-pdfs")
async def process_pdfs_endpoint(pdf_path: str = "./data/pdfs"):
    """Process PDFs and add to vector store"""
    try:
        from rag_core import process_pdfs, split_documents

        documents = process_pdfs(pdf_path)
        if not documents:
            raise HTTPException(status_code=404, detail="No PDF files found")

        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)

        return {
            "message": "PDFs processed successfully",
            "documents": len(documents),
            "chunks": len(chunks),
            "total_in_store": vector_store.get_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)