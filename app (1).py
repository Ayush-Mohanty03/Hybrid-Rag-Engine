import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Chat App", page_icon="📚")

st.title("📚 RAG Chat App")

# Sidebar for configuration
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=['pdf', 'txt'], accept_multiple_files=True)
    
    top_k = st.slider("Select Top-K documents to retrieve", min_value=1, max_value=15, value=5)
    
    st.header("Text Chunking Settings")
    chunk_size = st.slider("Chunk Size (characters)", min_value=500, max_value=5000, value=3000, step=100, help="Larger chunk sizes keep list items and sections together.")
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=1000, value=600, step=50, help="Overlapping characters between adjacent chunks to maintain context.")
    
    process_button = st.button("Process Documents")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_documents(files, chunk_size, chunk_overlap):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please add your Groq API key (GROQ_API_KEY) to the .env file.")
        return None
    
    documents = []
    with st.spinner("Reading files..."):
        for file in files:
            file_extension = file.name.split('.')[-1].lower()
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
                
            try:
                if file_extension == 'pdf':
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                elif file_extension == 'txt':
                    loader = TextLoader(temp_file_path)
                    docs = loader.load()
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
                    continue
                for doc in docs:
                    doc.metadata["source_filename"] = file.name
                    
                documents.extend(docs)
            finally:
                os.remove(temp_file_path)
                
    if not documents:
        st.warning("No documents loaded.")
        return None
        
    with st.spinner("Chunking documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        
    with st.spinner("Creating embeddings and storing in vector database..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        
       
        import shutil
        if os.path.exists(persist_directory):
            try:
                
                old_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                old_db.delete_collection()
            except Exception:
                pass
            try:
                
                shutil.rmtree(persist_directory)
            except Exception:
                pass
                
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
    return vector_store

if process_button and uploaded_files:
    vector_store = process_documents(uploaded_files, chunk_size, chunk_overlap)
    if vector_store:
        st.session_state.vector_store = vector_store


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
        
        
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "\n\n"
            "{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            source_documents = response.get("context", [])
            
            st.markdown(answer)
            
            if source_documents:
                with st.expander("Sources"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source_filename', 'Unknown')}")
                        st.markdown(f"**Content Snippet:** {doc.page_content[:200]}...")
                        st.markdown("---")
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
