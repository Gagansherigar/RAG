import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import uuid

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# ============= DOCUMENT PROCESSING =============
def process_pdfs(pdf_directory: str) -> List:
    """Load all PDFs from directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    return all_documents


def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# ============= EMBEDDINGS =============
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)

    def get_embeddings_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# ============= VECTOR STORE =============
class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents",
                 persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF documents for RAG"}
        )

    def add_documents(self, documents: List, embeddings: np.ndarray):
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [dict(doc.metadata, doc_index=i, content_length=len(doc.page_content))
                     for i, doc in enumerate(documents)]
        documents_text = [doc.page_content for doc in documents]
        embeddings_list = [emb.tolist() for emb in embeddings]

        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings_list,
            documents=documents_text
        )

    def get_count(self) -> int:
        return self.collection.count()


# ============= RETRIEVER =============
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, content, metadata, distance) in enumerate(
                    zip(results["ids"][0], results["documents"][0],
                        results["metadatas"][0], results["distances"][0])
            ):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })

        return retrieved_docs


# ============= RAG FUNCTIONS =============
def rag_simple(query: str, retriever: RAGRetriever, llm, top_k: int = 3) -> str:
    """Simple RAG"""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

    prompt = f"""Use the following context to answer the question concisely.

Context: {context}

Question: {query}

Answer:"""

    response = llm.invoke([prompt])
    return response.content


def rag_advanced(query: str, retriever: RAGRetriever, llm, top_k: int = 5,
                 min_score: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
    """Advanced RAG with sources"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            'answer': 'No relevant context found.',
            'sources': [],
            'confidence': 0.0,
            'context': '' if return_context else None
        }

    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', 'unknown'),
        'page': str(doc['metadata'].get('page', 'unknown')),
        'score': float(doc['similarity_score']),
        'preview': doc['content'][:300] + '...' if len(doc['content']) > 300 else doc['content']
    } for doc in results]

    confidence = float(max([doc['similarity_score'] for doc in results]))

    prompt = f"""Use the following context to answer the question concisely.

Context: {context}

Question: {query}

Answer:"""

    response = llm.invoke([prompt])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context
    else:
        output['context'] = None

    return output


class AdvancedRAGPipeline:
    """Pipeline with history"""

    def __init__(self, retriever: RAGRetriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2,
              stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            answer = "No relevant context found."
            sources = []
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', 'unknown'),
                'page': str(doc['metadata'].get('page', 'unknown')),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]

            prompt = f"""Use the following context to answer the question concisely.

Context: {context}

Question: {question}

Answer:"""

            response = self.llm.invoke([prompt])
            answer = response.content

        # Add citations
        citations = [f"[{i + 1}] {src['source']} (page {src['page']})"
                     for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Summary
        summary = None
        if summarize and answer and answer != "No relevant context found.":
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

    def clear_history(self):
        self.history.clear()


# ============= INITIALIZATION =============
def initialize_rag(pdf_path: str = "./data/pdfs"):
    """Initialize entire RAG system"""
    print("Initializing RAG system...")

    # Setup components first
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    # Check if vector store is empty
    existing_count = vector_store.get_count()

    if existing_count == 0:
        print("Vector store is empty. Processing PDFs...")

        # Check if PDF directory exists
        if not os.path.exists(pdf_path):
            print(f"Creating PDF directory: {pdf_path}")
            os.makedirs(pdf_path, exist_ok=True)
            print(f"⚠️  Please add PDF files to {pdf_path} and restart")
        else:
            # Process documents
            documents = process_pdfs(pdf_path)

            if len(documents) == 0:
                print(f"⚠️  No PDF files found in {pdf_path}")
                print("   Add PDFs and restart, or use the API to process documents later")
            else:
                chunks = split_documents(documents)

                # Generate and store embeddings
                texts = [doc.page_content for doc in chunks]
                embeddings = embedding_manager.generate_embeddings(texts)
                vector_store.add_documents(chunks, embeddings)
                print(f"✓ Processed and stored {len(chunks)} document chunks")
    else:
        print(f"✓ Using existing vector store with {existing_count} documents")

    # Create retriever
    retriever = RAGRetriever(vector_store, embedding_manager)

    # Create LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )

    # Create pipeline
    pipeline = AdvancedRAGPipeline(retriever, llm)

    print(f"✓ RAG initialized with {vector_store.get_count()} documents")

    return retriever, llm, pipeline, embedding_manager, vector_store