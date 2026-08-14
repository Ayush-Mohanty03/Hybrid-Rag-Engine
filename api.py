import os
import shutil
import tempfile
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG Pipeline with Hybrid Search Over Internal Docs"

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate

from app import (
    RAW_DOCS_DIR,
    get_embeddings,
    get_cross_encoder,
    load_all_documents,
    chunk_documents,
    format_numbered_context,
    verify_citations,
    calculate_confidence_score,
    generate_structured_idk_response
)

app = FastAPI(
    title="End-to-End Hybrid RAG Engine API",
    version="1.0.0",
    description="Production-grade RAG REST API with Grounded Citations, Verification, and Multi-Metric Confidence Scoring."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "vector_store": None,
    "bm25_retriever": None
}

def get_or_load_vector_store():
    if state["vector_store"] is None:
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        embeddings = get_embeddings()
        dim = len(embeddings.embed_query("test"))
        col_name = f"rag_collection_{dim}"
        if os.path.exists(persist_directory):
            try:
                state["vector_store"] = Chroma(
                    collection_name=col_name,
                    embedding_function=embeddings,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
    return state["vector_store"]

class AskRequest(BaseModel):
    question: str = Field(..., example="What is the primary vision of LIC?")
    top_k: int = Field(default=5, ge=1, le=20, example=5)
    strategy: str = Field(default="Header-Aware Splitting", example="Header-Aware Splitting")

class AskResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    retrieval_score: float
    citation_coverage: float
    completeness_score: float
    source_documents_count: int

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Hybrid RAG Engine API",
        "version": "1.0.0"
    }

@app.get("/v1/documents", tags=["Documents"])
def list_documents():
    if not os.path.exists(RAW_DOCS_DIR):
        return {"documents": [], "total": 0}
    files = [f for f in os.listdir(RAW_DOCS_DIR) if os.path.isfile(os.path.join(RAW_DOCS_DIR, f))]
    return {"documents": files, "total": len(files)}

@app.post("/v1/ingest", tags=["Ingest"])
async def ingest_document(file: UploadFile = File(...)):
    allowed = ["pdf", "txt", "md", "html", "htm"]
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File extension .{ext} not supported. Allowed: {allowed}")
        
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DOCS_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "filename": file.filename,
        "message": f"Successfully uploaded {file.filename} to raw_documents storage."
    }

@app.post("/v1/ask", response_model=AskResponse, tags=["Query"])
def ask_question(request: AskRequest):
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured in .env file.")
        
    vector_store = get_or_load_vector_store()
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No vector store found. Please ingest documents first.")
        
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
    initial_k = 20
    cross_enc = get_cross_encoder()
    
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": initial_k})
    
    if state["bm25_retriever"] is not None:
        state["bm25_retriever"].k = initial_k
        hybrid_base = EnsembleRetriever(
            retrievers=[state["bm25_retriever"], dense_retriever],
            weights=[0.3, 0.7]
        )
    else:
        hybrid_base = dense_retriever

    compressor = CrossEncoderReranker(model=cross_enc, top_n=request.top_k)
    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_base
    )
    
    source_docs = hybrid_retriever.invoke(request.question)
    formatted_ctx = format_numbered_context(source_docs)
    
    system_prompt = (
        "You are a strict, factual assistant for question-answering tasks.\n"
        "Use ONLY the following numbered context blocks to answer the user's question.\n"
        "Rules:\n"
        "1. Every factual claim MUST end with a bracketed citation, e.g. [1].\n"
        "2. Do NOT use outside knowledge.\n"
        "3. If context lacks details, state: 'The provided context does not contain enough information'.\n\n"
        "Context:\n{context}"
    )
    
    pt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    msgs = pt.format_messages(context=formatted_ctx, input=request.question)
    raw_res = llm.invoke(msgs)
    answer = raw_res.content if hasattr(raw_res, 'content') else str(raw_res)
    
    vers = verify_citations(answer, source_docs, llm)
    c_conf, r_conf, cov_s, comp_s = calculate_confidence_score(source_docs, request.question, answer, vers, llm)
    is_low = c_conf < 40.0 or "does not contain enough information" in answer.lower()
    final_ans = generate_structured_idk_response(request.question, source_docs) if is_low else answer
    
    return AskResponse(
        question=request.question,
        answer=final_ans,
        confidence_score=c_conf,
        retrieval_score=r_conf,
        citation_coverage=cov_s,
        completeness_score=comp_s,
        source_documents_count=len(source_docs)
    )
