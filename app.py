import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama


# Ustawienia strony
st.title("Aplikacja z LlamaIndex")

# Załaduj dokumenty
documents = SimpleDirectoryReader('./data/').load_data()



db = chromadb.PersistentClient(path="./ustawy")
chroma_collection = db.get_or_create_collection("pomoc_ukrainie")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Utwórz pipeline do przetwarzania dokumentów
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store
)

# Przetwórz dokumenty
nodes = pipeline.run(documents)

# Utwórz indeks
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Utwórz silnik zapytań
llm = Ollama(model="qwen2:7b")
query_engine = index.as_query_engine(llm=llm)

# Wyświetl historię wiadomości
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wejście użytkownika
if input := st.text_input("Zadaj pytanie"):
    st.session_state.messages.append({"role": "user", "content": input})
    
    # Wygeneruj odpowiedź
    try:
        response = query_engine.query(input)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Błąd: {e}")

# Wyświetl wiadomości
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"Użytkownik: {message['content']}")
    else:
        st.write(f"Asystent: {message['content']}")
