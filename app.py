import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

# Ustawienia strony
st.set_page_config(page_title="LegitCheck", page_icon="⚖️")
st.title("LegitCheck")

db = chromadb.PersistentClient(path="./ustawy_BAAI")
chroma_collection = db.get_or_create_collection("pomoc_ukrainie")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Utwórz pipeline do przetwarzania dokumentów
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    vector_store=vector_store
)

# Utwórz indeks
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Utwórz silnik zapytań
llm = Ollama(model="qwen2:7b")
query_engine = index.as_query_engine(llm=llm)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Zadaj mi pytanie..."}]

# Display chat messages & custom avatar
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="⚖️"):  # zmieniamy ikonę robocika na sowę
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="⚖️"):  # tu też zmieniamy ikonę
        with st.spinner("Czekaj, odpowiedź jest generowana.."):
            response = query_engine.query(input) 
            st.write(response.response)  # Tylko odpowiedź
            
    message = {"role": "assistant", "content": response.response}  # poprawka, żeby content był tekstem
    st.session_state.messages.append(message)


# Sidebar FAQ
with st.sidebar:
    st.header("FAQ")
    example_questions = [
        "Od kogo nie można pobierać odcisków palców?",
        "Jakie są procedury legalizacji pobytu?",
        "Kto może ubiegać się o pomoc humanitarną?",
        "Jakie dokumenty są wymagane do rejestracji?",
        "Kiedy można odmówić udzielenia pomocy?"
    ]
    
    for question in example_questions:
        st.write(question)