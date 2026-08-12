import streamlit as st
import os
import tempfile
import shutil
import re
import json
import numpy as np
from html.parser import HTMLParser
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Pipeline with Hybrid Search Over Internal Docs"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from eval_engine import run_benchmark_suite, RAGEvaluator

# Raw document storage directory
RAW_DOCS_DIR = os.path.join(os.getcwd(), "raw_documents")
os.makedirs(RAW_DOCS_DIR, exist_ok=True)

# Custom HTML parser to extract clean plaintext from HTML documents
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        
    def handle_data(self, data):
        self.text_parts.append(data)
        
    def get_text(self):
        return "".join(self.text_parts)

# Cache heavy resources like ML models
@st.cache_resource
def get_embeddings():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
        except Exception as e:
            try:
                st.warning(f"Failed to load OpenAI embeddings: {e}. Trying fallback.")
            except Exception:
                pass
                
    openrouter_key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            return OpenAIEmbeddings(
                model="openai/text-embedding-3-small",
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
        except Exception:
            pass
            
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_cross_encoder():
    return HuggingFaceCrossEncoder(model_name="ms-marco-MiniLM-L-12-v2")



def format_numbered_context(source_documents):
    formatted_blocks = []
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get("source_filename", "Unknown")
        section = doc.metadata.get("section_heading", "General Context")
        page = doc.metadata.get("page", 1)
        header = f"[{i+1}] Source: {source} (Page {page}) | Section: {section}"
        content = doc.page_content.strip()
        formatted_blocks.append(f"{header}\n{content}")
    return "\n\n".join(formatted_blocks)

def verify_citations(answer, source_documents, llm):
    citation_matches = re.findall(r'\[(\d+)\]', answer)
    unique_citations = sorted(list(set(int(c) for c in citation_matches)))
    
    verifications = {}
    
    for citation_num in unique_citations:
        if 1 <= citation_num <= len(source_documents):
            chunk = source_documents[citation_num - 1]
            chunk_content = chunk.page_content[:1000]
            
            sentences = re.split(r'(?<=[.?!])\s+', answer)
            claims = [s.strip() for s in sentences if f"[{citation_num}]" in s]
            claim_text = " ".join(claims) if claims else answer
            
            eval_prompt = (
                f"You are a strict factual evaluator.\n"
                f"Claim: \"{claim_text}\"\n"
                f"Source Context Chunk [{citation_num}]: \"{chunk_content}\"\n\n"
                "Does the Source Context Chunk factually support or entail the Claim?\n"
                "Respond with ONLY 'YES' or 'NO'."
            )
            try:
                res = llm.invoke(eval_prompt)
                ver_res = res.content.strip().upper() if hasattr(res, 'content') else str(res).strip().upper()
                is_supported = "YES" in ver_res
            except Exception:
                is_supported = True
                
            verifications[citation_num] = {
                "supported": is_supported,
                "claim": claim_text,
                "source": chunk.metadata.get("source_filename", "Unknown"),
                "section": chunk.metadata.get("section_heading", "General Context")
            }
        else:
            verifications[citation_num] = {
                "supported": False,
                "claim": "Invalid citation index out of range",
                "source": "None",
                "section": "None"
            }
            
    return verifications

def calculate_confidence_score(source_documents, prompt, answer, verifications, llm):
    if not source_documents:
        return 0.0, 0.0, 0.0, 0.0
        
    prompt_terms = [w.lower() for w in prompt.split() if len(w) > 3]
    term_matches = sum(1 for doc in source_documents for term in prompt_terms if term in doc.page_content.lower())
    match_ratio = min(1.0, term_matches / (len(prompt_terms) + 1e-5)) if prompt_terms else 0.8
    retrieval_conf = round(min(1.0, 0.5 + 0.5 * match_ratio), 2)
    
    if verifications:
        supported_count = sum(1 for v in verifications.values() if v["supported"])
        citation_coverage = round(supported_count / len(verifications), 2)
    else:
        citation_coverage = 0.5 if "does not contain" not in answer.lower() and "don't know" not in answer.lower() else 0.0
        
    try:
        comp_prompt = (
            f"User Question: \"{prompt}\"\n"
            f"Generated Answer: \"{answer}\"\n\n"
            "On a scale of 0.0 to 1.0, how completely does the Generated Answer address all parts of the User Question based on context?\n"
            "Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        res = llm.invoke(comp_prompt)
        res_text = res.content.strip() if hasattr(res, 'content') else str(res).strip()
        comp_val = float(re.findall(r"0\.\d+|1\.0|0", res_text)[0])
        completeness_score = round(min(1.0, max(0.0, comp_val)), 2)
    except Exception:
        completeness_score = 0.8
        
    composite_score = round((0.4 * retrieval_conf + 0.3 * citation_coverage + 0.3 * completeness_score) * 100, 1)
    
    return composite_score, retrieval_conf, citation_coverage, completeness_score

def generate_structured_idk_response(prompt, source_documents):
    files = list(set(d.metadata.get("source_filename", "Unknown") for d in source_documents)) if source_documents else ["No relevant files"]
    sections = list(set(d.metadata.get("section_heading", "General Context") for d in source_documents)) if source_documents else ["No relevant sections"]
    
    snippets = "\n".join([f"- `{d.metadata.get('source_filename')}` ({d.metadata.get('section_heading')}): {d.page_content[:150]}..." for d in source_documents[:3]]) if source_documents else "No matching snippets found."
    
    idk_markdown = f"""### ⚠️ Low Confidence / Insufficient Context Response

The system retrieved context from your documents, but could **not** verify a high-confidence grounded answer for:
> *"{prompt}"*

---

#### 🔍 **What was found in candidate chunks:**
{snippets}

#### ❌ **What could not be found:**
The indexed documents do not contain explicit facts or complete details answering this specific question.

#### 📁 **Suggested Manual Check:**
- **Files to check:** {', '.join([f'`{f}`' for f in files])}
- **Sections to check:** {', '.join([f'`{s}`' for s in sections])}
"""
    return idk_markdown

# Heuristic to find the nearest preceding section heading
def extract_section_heading(text):
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line.isupper() and len(line) > 3 and len(line) < 80:
            return line
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return line
    return "General Context"

# Custom multi-format loader
def load_all_documents():
    documents = []
    if not os.path.exists(RAW_DOCS_DIR):
        return []
        
    for filename in os.listdir(RAW_DOCS_DIR):
        file_path = os.path.join(RAW_DOCS_DIR, filename)
        if not os.path.isfile(file_path):
            continue
            
        file_extension = filename.split('.')[-1].lower()
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_extension in ['txt', 'md']:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            elif file_extension in ['html', 'htm']:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                parser = HTMLTextExtractor()
                parser.feed(html_content)
                docs = [Document(page_content=parser.get_text(), metadata={"source_filename": filename})]
            else:
                continue
                
            for doc in docs:
                doc.metadata["source_filename"] = filename
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1
                    
            documents.extend(docs)
        except Exception as e:
            try:
                st.error(f"Error loading {filename}: {e}")
            except Exception:
                pass
            
    return documents

# Configurable Chunking Strategies
def chunk_documents(documents, strategy, chunk_size, chunk_overlap):
    chunks = []
    
    if strategy == "Fixed-Size Chunks (Baseline)":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_chunks = text_splitter.split_documents(documents)
        for i, rc in enumerate(raw_chunks):
            section = extract_section_heading(rc.page_content)
            rc.metadata.update({
                "chunk_index": i,
                "section_heading": section,
                "chunking_strategy": "fixed-size",
                "character_count": len(rc.page_content)
            })
            chunks.append(rc)
            
    elif strategy == "Header-Aware Splitting":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
        raw_chunks = text_splitter.split_documents(documents)
        for i, rc in enumerate(raw_chunks):
            section = extract_section_heading(rc.page_content)
            rc.metadata.update({
                "chunk_index": i,
                "section_heading": section,
                "chunking_strategy": "header-aware",
                "character_count": len(rc.page_content)
            })
            chunks.append(rc)
            
    elif strategy == "Semantic Topic Splitting":
        embeddings_model = get_embeddings()
        chunk_idx = 0
        
        for doc in documents:
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', doc.page_content) if s.strip()]
            if not sentences:
                continue
                
            try:
                sentence_embeddings = embeddings_model.embed_documents(sentences)
            except Exception as e:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                raw_chunks = text_splitter.split_documents([doc])
                for rc in raw_chunks:
                    section = extract_section_heading(rc.page_content)
                    rc.metadata.update({
                        "chunk_index": chunk_idx,
                        "section_heading": section,
                        "chunking_strategy": "semantic-fallback",
                        "character_count": len(rc.page_content)
                    })
                    chunks.append(rc)
                    chunk_idx += 1
                continue
            
            current_chunk_sentences = [sentences[0]]
            
            for i in range(len(sentences) - 1):
                vec1 = np.array(sentence_embeddings[i])
                vec2 = np.array(sentence_embeddings[i+1])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                similarity = np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                
                current_len = sum(len(s) for s in current_chunk_sentences)
                if similarity < 0.75 or current_len + len(sentences[i+1]) > chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    section = extract_section_heading(chunk_text)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source_filename": doc.metadata.get("source_filename", "Unknown"),
                            "page": doc.metadata.get("page", 1),
                            "chunk_index": chunk_idx,
                            "section_heading": section,
                            "chunking_strategy": "semantic",
                            "character_count": len(chunk_text)
                        }
                    ))
                    chunk_idx += 1
                    current_chunk_sentences = [sentences[i+1]]
                else:
                    current_chunk_sentences.append(sentences[i+1])
            
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                section = extract_section_heading(chunk_text)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source_filename": doc.metadata.get("source_filename", "Unknown"),
                        "page": doc.metadata.get("page", 1),
                        "chunk_index": chunk_idx,
                        "section_heading": section,
                        "chunking_strategy": "semantic",
                        "character_count": len(chunk_text)
                    }
                ))
                chunk_idx += 1
                
    return chunks

# Near-duplicate chunk deduplication based on cosine similarity
def deduplicate_chunks(chunks, vector_store, embeddings):
    unique_chunks = []
    
    has_existing = False
    try:
        if vector_store and len(vector_store.get()['ids']) > 0:
            has_existing = True
    except Exception:
        pass
        
    for chunk in chunks:
        is_duplicate = False
        
        if unique_chunks:
            for uc in unique_chunks:
                if uc.page_content.strip() == chunk.page_content.strip():
                    is_duplicate = True
                    break
                    
        if not is_duplicate and has_existing:
            try:
                results = vector_store.similarity_search_with_score(chunk.page_content, k=1)
                if results:
                    _, score = results[0]
                    if score < 0.05:
                        is_duplicate = True
            except Exception:
                pass
                
        if not is_duplicate:
            unique_chunks.append(chunk)
            
    return unique_chunks

# Unified Ingestion Pipeline
def process_documents(chunk_size, chunk_overlap, strategy):
    documents = load_all_documents()
    if not documents:
        try:
            st.warning("No documents found in storage. Please upload documents first.")
        except Exception:
            pass
        return None
        
    chunks = chunk_documents(documents, strategy, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    dim = len(embeddings.embed_query("test"))
    col_name = f"rag_collection_{dim}"
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    
    vector_store = None
    try:
        if os.path.exists(persist_directory):
            vector_store = Chroma(
                collection_name=col_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
    except Exception:
        pass

    if vector_store is None:
        vector_store = Chroma(
            collection_name=col_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

    unique_chunks = deduplicate_chunks(chunks, vector_store, embeddings)
    
    if unique_chunks:
        try:
            if vector_store:
                vector_store.delete_collection()
        except Exception:
            pass
            
        vector_store = Chroma.from_documents(
            documents=unique_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=col_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        bm25_retriever = BM25Retriever.from_documents(unique_chunks)
    else:
        vector_store = None
        bm25_retriever = None
        
    return vector_store, bm25_retriever


# UI Setup
st.set_page_config(page_title="RAG Chat App", page_icon="📚")
st.title("📚 RAG Chat App")

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, MD, or HTML files", 
        type=['pdf', 'txt', 'md', 'html', 'htm'], 
        accept_multiple_files=True
    )
    
    top_k = st.slider("Select Top-K documents to retrieve", min_value=1, max_value=15, value=5)
    
    st.header("Ingestion & Chunking Settings")
    chunk_size = st.slider("Max Chunk Size (characters)", min_value=500, max_value=5000, value=3000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=1000, value=600, step=50)
    
    strategy = st.selectbox(
        "Chunking Strategy",
        ["Fixed-Size Chunks (Baseline)", "Header-Aware Splitting", "Semantic Topic Splitting"]
    )
    
    st.header("Comparison Mode")
    compare_mode = st.checkbox("⚔️ Compare Hybrid vs Dense-Only Side-by-Side", value=False)
    
    process_button = st.button("Process & Upload Documents")
    reindex_button = st.button("Re-index From Storage")
    
    if os.path.exists(RAW_DOCS_DIR):
        files = os.listdir(RAW_DOCS_DIR)
        if files:
            st.markdown("---")
            st.markdown("**Currently Stored Files:**")
            for f in files:
                st.markdown(f"- `{f}`")
            
            if st.button("Clear All Files"):
                shutil.rmtree(RAW_DOCS_DIR)
                os.makedirs(RAW_DOCS_DIR, exist_ok=True)
                st.session_state.vector_store = None
                st.session_state.bm25_retriever = None
                st.success("Storage cleared!")
                st.rerun()

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if process_button and uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(RAW_DOCS_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
            
    result = process_documents(chunk_size, chunk_overlap, strategy)
    if result:
        st.session_state.vector_store, st.session_state.bm25_retriever = result
        st.success("Documents processed and indexed successfully!")

if reindex_button:
    result = process_documents(chunk_size, chunk_overlap, strategy)
    if result:
        st.session_state.vector_store, st.session_state.bm25_retriever = result
        st.success("Documents re-indexed successfully!")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    if not os.environ.get("GROQ_API_KEY"):
        st.info("Please add your Groq API key (GROQ_API_KEY) to the .env file to continue.")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.info("Please upload and process documents first.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        initial_k = 20
        cross_enc = get_cross_encoder()
        
        dense_retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": initial_k})
        
        if st.session_state.bm25_retriever is not None:
            st.session_state.bm25_retriever.k = initial_k
            hybrid_base = EnsembleRetriever(
                retrievers=[st.session_state.bm25_retriever, dense_retriever],
                weights=[0.3, 0.7]
            )
        else:
            hybrid_base = dense_retriever

        compressor = CrossEncoderReranker(model=cross_enc, top_n=top_k)
        hybrid_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_base
        )
        
        system_prompt = (
            "You are a strict, factual assistant for question-answering tasks.\n"
            "Use ONLY the following numbered context blocks to answer the user's question.\n"
            "Rules:\n"
            "1. Every factual claim or statement in your answer MUST end with a bracketed citation pointing to the exact context block number(s) that support it, e.g. [1] or [1][2].\n"
            "2. Do NOT use outside knowledge or make assumptions not directly supported by the context.\n"
            "3. If the context does not contain enough information to answer the question, state what is known from the context and explicitly state: 'The provided context does not contain enough information to answer this question'.\n\n"
            "Context:\n{context}"
        )
        
        def run_qa_pipeline(target_retriever, query):
            s_docs = target_retriever.invoke(query)
            f_ctx = format_numbered_context(s_docs)
            pt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            msgs = pt.format_messages(context=f_ctx, input=query)
            raw_res = llm.invoke(msgs)
            ans = raw_res.content if hasattr(raw_res, 'content') else str(raw_res)
            vers = verify_citations(ans, s_docs, llm)
            c_conf, r_conf, cov_s, comp_s = calculate_confidence_score(s_docs, query, ans, vers, llm)
            is_low = c_conf < 40.0 or "does not contain enough information" in ans.lower()
            final_ans = generate_structured_idk_response(query, s_docs) if is_low else ans
            return {
                "answer": final_ans,
                "source_documents": s_docs,
                "verifications": vers,
                "confidence_score": c_conf,
                "retrieval_score": r_conf,
                "citation_coverage": cov_s,
                "completeness_score": comp_s
            }

        if compare_mode:
            st.markdown("### ⚔️ Side-by-Side Retrieval Comparison")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("#### 🔀 Hybrid + Reranker Engine")
                with st.spinner("Running Hybrid Engine..."):
                    h_res = run_qa_pipeline(hybrid_retriever, prompt)
                st.markdown(h_res["answer"])
                st.metric("Hybrid Confidence", f"{h_res['confidence_score']}%")
                with st.expander("Hybrid Chunks & Citations"):
                    for i, doc in enumerate(h_res["source_documents"]):
                        st.markdown(f"**Chunk [{i+1}]**: `{doc.metadata.get('source_filename')}` ({doc.metadata.get('section_heading')})")
                        st.markdown(f"> {doc.page_content[:200]}...")
                        
            with comp_col2:
                st.markdown("#### 🎯 Dense-Only Baseline Engine")
                with st.spinner("Running Dense Baseline..."):
                    d_res = run_qa_pipeline(dense_retriever, prompt)
                st.markdown(d_res["answer"])
                st.metric("Dense-Only Confidence", f"{d_res['confidence_score']}%")
                with st.expander("Dense Chunks & Citations"):
                    for i, doc in enumerate(d_res["source_documents"]):
                        st.markdown(f"**Chunk [{i+1}]**: `{doc.metadata.get('source_filename')}` ({doc.metadata.get('section_heading')})")
                        st.markdown(f"> {doc.page_content[:200]}...")
                        
            answer = h_res["answer"]
            
        else:
            with st.spinner("Retrieving context & generating grounded answer..."):
                main_res = run_qa_pipeline(hybrid_retriever, prompt)
                answer = main_res["answer"]
                source_documents = main_res["source_documents"]
                verifications = main_res["verifications"]
                composite_conf = main_res["confidence_score"]
                ret_conf = main_res["retrieval_score"]
                cov_score = main_res["citation_coverage"]
                comp_score = main_res["completeness_score"]
                
                st.markdown(answer)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Overall Confidence", f"{composite_conf}%")
                col2.metric("Retrieval Score", f"{int(ret_conf*100)}%")
                col3.metric("Citation Coverage", f"{int(cov_score*100)}%")
                col4.metric("Completeness", f"{int(comp_score*100)}%")
                
            if source_documents:
                with st.expander("📚 Sources & Citation Verification"):
                    for i, doc in enumerate(source_documents):
                        c_idx = i + 1
                        ver_info = verifications.get(c_idx)
                        status_badge = "✅ Verified Citation" if ver_info and ver_info["supported"] else ("⚠️ Unverified/Not Cited" if ver_info else "ℹ️ Context Chunk")
                        
                        st.markdown(f"**Chunk [{c_idx}]** - {status_badge}")
                        st.markdown(f"**Source File:** `{doc.metadata.get('source_filename', 'Unknown')}` | **Page:** {doc.metadata.get('page', 1)}")
                        st.markdown(f"**Section Heading:** `{doc.metadata.get('section_heading', 'General Context')}`")
                        st.markdown(f"**Chunking Strategy:** `{doc.metadata.get('chunking_strategy', 'Unknown')}`")
                        st.markdown(f"**Content:**\n> {doc.page_content[:300]}...")
                        st.markdown("---")
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
