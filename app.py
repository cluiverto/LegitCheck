import streamlit as st
import logging
from typing import List, Dict, Optional
import requests
from elasticsearch import Elasticsearch
import os, time
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Klasa reprezentująca fragment dokumentu"""
    content: str
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None


@dataclass
class LegalRAGConfig:
    """Konfiguracja systemu RAG"""
    elasticsearch_host: str
    elasticsearch_index: str
    embedding_service_url: str
    reranking_service_url: str
    ollama_host: str
    model_name: str

    retrieval_top_k: int = 50
    rerank_top_k: int = 3
    max_context_length: int = 4000
    use_reranking: bool = True
    embedding_timeout: int = 120
    embedding_retries: int = 2


class EmbeddingClient:
    """Klient do serwisu embedingów"""

    def __init__(self, service_url: str, timeout: int = 120, retries: int = 2):
        self.service_url = service_url
        self.timeout = timeout
        self.retries = retries
        self.embedding_cache: Dict[str, List[float]] = {}
        logger.info(f"EmbeddingClient: {service_url}")

    def get_embedding(self, text: str) -> List[float]:
        """Pobiera embedding"""
        text_hash = str(hash(text))  #hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    f"{self.service_url}/embed",
                    json={"text": text},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    if embedding:
                        self.embedding_cache[text_hash] = embedding
                        return embedding
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt}/{self.retries} failed: {e}")
                if attempt < self.retries:
                    time.sleep(2 ** (attempt - 1))

        return []


class RerankingClient:
    """Klient do serwisu rerankingu"""

    def __init__(self, service_url: str, timeout: int = 300):
        self.service_url = service_url
        self.timeout = timeout
        logger.info(f"RerankingClient: {service_url}")

    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> tuple:
        """Reranking dokumentów"""
        try:
            response = requests.post(
                f"{self.service_url}/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": top_k
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("scores", []), result.get("ranked_indices", [])
        except Exception as e:
            logger.error(f"Reranking error: {e}")

        return [], []


class ElasticsearchRetrievalClient:
    """Klient Elasticsearch"""

    def __init__(self, host: str, index_name: str, embedding_client: EmbeddingClient):
        self.es = Elasticsearch([host])
        self.index_name = index_name
        self.embedding_client = embedding_client

    def search_with_embeddings(self, query: str, top_k: int = 50) -> List[DocumentChunk]:
        """Wyszukiwanie semantyczne"""
        try:
            query_embedding = self.embedding_client.get_embedding(query)

            if not query_embedding:
                return self._search_bm25(query, top_k)

            search_body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, '_embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "_source": ["text", "title", "article", "paragraph", "id"]
            }

            response = self.es.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response['hits']['hits']:
                source_data = hit['_source']

                source_parts = []
                if 'article' in source_data and source_data['article']:
                    source_parts.append(f"art. {source_data['article']}")
                if 'paragraph' in source_data and source_data['paragraph']:
                    source_parts.append(f"§ {source_data['paragraph']}")
                if 'id' in source_data and source_data['id']:
                    source_parts.append(f"{source_data['id']}")

                source = " - ".join(source_parts) if source_parts else "Nieznane źródło"

                chunk = DocumentChunk(
                    content=source_data.get('text', ''),
                    retrieval_score=hit['_score'],
                    chunk_id=hit['_id'],
                    source=source
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._search_bm25(query, top_k)

    def _search_bm25(self, query: str, top_k: int = 50) -> List[DocumentChunk]:
        """Fallback BM25"""
        try:
            search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                        "fuzziness": "AUTO"
                    }
                },
                "_source": ["text", "title", "article", "paragraph"]
            }

            response = self.es.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response['hits']['hits']:
                source_data = hit['_source']
                source_parts = []
                if 'title' in source_data:
                    source_parts.append(source_data['title'])

                chunk = DocumentChunk(
                    content=source_data.get('text', ''),
                    retrieval_score=hit['_score'],
                    chunk_id=hit['_id'],
                    source=" - ".join(source_parts) if source_parts else "BM25"
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"BM25 error: {e}")
            return []


class OllamaClient:
    """Klient Ollama"""

    def __init__(self, host: str, model_name: str):
        self.host = host
        self.model_name = model_name

    def generate_response(self, context: str, question: str) -> str:
        """Generuje odpowiedź"""
        prompt = f"""Jesteś asystentem prawnym specjalizującym się w polskim prawie. Na podstawie podanych fragmentów dokumentów odpowiedz na pytanie użytkownika.



FRAGMENTY:
{context}



PYTANIE: {question}



INSTRUKCJE:
1. Odpowiadaj wyłącznie na podstawie podanych fragmentów
2. Fragmenty zostały wybrane jako NAJLEPSZE spośród wielu opcji
3. Jeśli pytanie dotyczy definicji, zacytuj dokładnie odpowiedni fragment
4. Używaj jasnego, zrozumiałego języka
5. Jeśli nie ma wystarczających informacji w podanych fragmentach, napisz to jasno
6. Nie dodawaj informacji spoza podanych fragmentów



ODPOWIEDŹ:"""

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=1200
            )

            if response.status_code == 200:
                return response.json().get('response', 'Brak odpowiedzi')

            return "Błąd generowania"
        except Exception as e:
            return f"Błąd: {str(e)}"


class LegalRAGSystem:
    """System RAG"""

    def __init__(self, config: LegalRAGConfig):
        self.config = config

        embedding_client = EmbeddingClient(
            config.embedding_service_url,
            timeout=config.embedding_timeout,
            retries=config.embedding_retries
        )

        self.es_client = ElasticsearchRetrievalClient(
            config.elasticsearch_host,
            config.elasticsearch_index,
            embedding_client
        )

        self.reranking_client = RerankingClient(
            config.reranking_service_url
        ) if config.use_reranking else None

        self.ollama_client = OllamaClient(config.ollama_host, config.model_name)

    def process_question(self, question: str) -> dict:
        """Pipeline przetwarzania z progress tracking"""

        # RETRIEVAL
        with st.spinner("🔍 Wyszukiwanie fragmentów..."):
            retrieved = self.es_client.search_with_embeddings(
                question,
                self.config.retrieval_top_k
            )

        if not retrieved:
            return {
                'answer': 'Brak dokumentów',
                'chunks': [],
                'confidence': 0.0,
                'pipeline_stats': {'retrieved': 0, 'final': 0, 'reranking_used': False}
            }

        st.success(f"✅ Znaleziono {len(retrieved)} fragmentów")

        # RERANKING
        final = retrieved
        reranking_used = False

        if self.config.use_reranking and self.reranking_client:
            with st.spinner("🎯 Reranking dokumentów..."):
                documents = [chunk.content for chunk in retrieved]
                scores, indices = self.reranking_client.rerank(
                    question,
                    documents,
                    self.config.rerank_top_k
                )

                if indices:
                    final = [retrieved[i] for i in indices]
                    for i, idx in enumerate(indices):
                        final[i].rerank_score = float(scores[i])
                    reranking_used = True
                    st.success(f"✅ Reranking: wybrano {len(final)} najlepszych")

        if not reranking_used and len(final) > self.config.rerank_top_k:
            final = final[:self.config.rerank_top_k]

        # GENEROWANIE
        context = self._prepare_context(final)
        with st.spinner("🧠 Generowanie odpowiedzi..."):
            answer = self.ollama_client.generate_response(context, question)

        return {
            'answer': answer,
            'chunks': final,
            'confidence': self._calculate_confidence(final),
            'pipeline_stats': {
                'retrieved': len(retrieved),
                'final': len(final),
                'reranking_used': reranking_used
            }
        }

    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        parts = []
        length = 0

        for i, chunk in enumerate(chunks, 1):
            score = chunk.rerank_score if chunk.rerank_score else chunk.retrieval_score
            frag = f"[{i}] {chunk.source} (score: {score:.3f})\n{chunk.content}\n"

            if length + len(frag) > self.config.max_context_length:
                break

            parts.append(frag)
            length += len(frag)

        return "\n".join(parts)

    def _calculate_confidence(self, chunks: List[DocumentChunk]) -> float:
        if not chunks:
            return 0.0
        scores = [c.rerank_score or c.retrieval_score for c in chunks]
        return sum(scores) / (len(scores) * max(scores)) if scores else 0.0


def search_phrase_in_index(es_client, index_name, phrase):
    query = {
        "query": {
            "match_phrase": {
                "content": phrase
            }
        }
    }

    try:
        res = es_client.search(index=index_name, body=query)
    except Exception as e:
        st.error(f"Błąd zapytania do Elasticsearch: {e}")
        return []

    results = []
    for hit in res['hits']['hits']:
        content = hit['_source']['content'].lower()
        count = content.count(phrase.lower())
        if count > 0:
            filename = hit['_source']['filename']
            if filename.endswith(".txt"):
                filename = filename[:-4]
            results.append({
                "document": filename,
                "count": count
            })

    results.sort(key=lambda x: x['count'], reverse=True)
    return results


def init_session_state():
    """Inicjalizuje session state"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def main():
    """Główna funkcja"""

    st.set_page_config(
        page_title="HDC Sandbox MWI",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Header
    st.title("⚖️ HDC Sandbox - MWI")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Konfiguracja")

        available_models = [
            "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0",  # default
            "SpeakLeash/bielik-1.5b-v3.0-instruct:Q8_0",
            "PRIHLOP/PLLuM:12b"
        ]

        model_name = st.selectbox(
            "Model LLM",
            options=available_models,
            index=0,
            help="Wybierz model LLM do generowania odpowiedzi"
        )
        use_reranking = True

        retrieval_top_k = st.slider("Retrieval top-K", 10, 100, 50)
        rerank_top_k = st.slider("Reranking top-K", 3, 20, 3)

        st.markdown("---")

        # Inicjalizacja
        if st.button("🔄 Inicjalizuj System", type="primary"):
            config = LegalRAGConfig(
                elasticsearch_host=os.getenv("ELASTICSEARCH_HOST"),
                elasticsearch_index=os.getenv("ES_INDEX"),
                embedding_service_url=os.getenv("EMBED_URL"),
                reranking_service_url=os.getenv("RERANK_URL"),
                ollama_host=os.getenv("OLLAMA_HOST"),
                model_name=model_name,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
                use_reranking=use_reranking
            )

            try:
                with st.spinner("Inicjalizacja..."):
                    st.session_state.rag_system = LegalRAGSystem(config)
                st.success("✅ System zainicjalizowany!")
            except Exception as e:
                st.error(f"❌ Błąd: {e}")

        # Status
        if st.session_state.rag_system:
            st.success("🟢 System gotowy")
        else:
            st.warning("🟡 System niezainicjalizowany")

        # Reset
        if st.button("🗑️ Wyczyść Historię"):
            st.session_state.chat_history = []
            st.rerun()

    # Główny interfejs
    col1, col2 = st.columns([2, 1])

    with col1:
        es = Elasticsearch(os.getenv("ELASTICSEARCH_HOST"))
        with st.expander("🔍 Wyszukiwarka fraz", expanded=False):
            phrase = st.text_input("Wpisz frazę do wyszukania")

            if phrase:
                results = search_phrase_in_index(es, 'ustawy', phrase)

                if results:
                    st.write(f"Liczba dokumentów zawierających frazę '{phrase}': {len(results)}")
                    for r in results:
                        st.write(f"- **{r['document']}** — liczba wystąpień: {r['count']}")
                else:
                    st.write(f"Brak dokumentów zawierających frazę '{phrase}'.")

        st.subheader("💬 Zadaj Pytanie")

        with st.form("question_form"):
            question = st.text_area(
                "Twoje pytanie:",
                height=100,
                placeholder="Np: Jaka jest definicja produktu leczniczego?"
            )

            submitted = st.form_submit_button("🔍 Zadaj Pytanie", type="primary")

            if submitted and question and st.session_state.rag_system:
                start_time = time.time()

                # Przetwarzanie
                result = st.session_state.rag_system.process_question(question)

                end_time = time.time()
                response_time = end_time - start_time

                # Historia
                st.session_state.chat_history.append({
                    'question': question,
                    'result': result,
                    'response_time': response_time,
                    'timestamp': time.strftime('%H:%M:%S')
                })

                st.rerun()

        # Historia
        if st.session_state.chat_history:
            st.subheader("📜 Historia Rozmów")

            for chat in reversed(st.session_state.chat_history):
                with st.expander(f"💬 {chat['timestamp']} - {chat['question'][:50]}..."):
                    st.markdown(f"**Pytanie:** {chat['question']}")

                    st.markdown("**Odpowiedź:**")
                    st.write(chat['result']['answer'])

                    # Metryki
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("⏱️ Czas", f"{chat['response_time']:.2f}s")
                    with col_m2:
                        st.metric("📈 Pewność", f"{chat['result']['confidence']:.0%}")
                    with col_m3:
                        stats = chat['result']['pipeline_stats']
                        st.metric("🎯 Reranking", "✅" if stats['reranking_used'] else "❌")


                    if chat['result']['chunks']:
                        st.markdown("**📚 Źródła:**")
                        for i, chunk in enumerate(chat['result']['chunks'], 1):
                            score = chunk.rerank_score if chunk.rerank_score else chunk.retrieval_score
                            score_type = "🎯 Rerank" if chunk.rerank_score else "📊 Retrieval" 

                            with st.container():
                                st.markdown(f"**{i}. {chunk.source}** ({score_type}: {score:.3f})")
                                preview = chunk.content
                                st.text(preview)
                                st.markdown("---")

if __name__ == "__main__":
    main()

                