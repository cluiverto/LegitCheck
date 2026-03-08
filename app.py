import json
import logging
from typing import List, Optional
import requests
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from dataclasses import dataclass
import os


# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Klasa reprezentująca fragment dokumentu"""
    content: str
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None


@dataclass
class LegalRAGConfig:
    """Konfiguracja systemu RAG z rerankingiem"""
    elasticsearch_host: str = "http://elasticsearch-no-ssl:9200"
    elasticsearch_index: str = "mwi4"
    embedding_service_url: str = "http://localhost:5000"  # tutaj URL do kontenera embeddingów
    ollama_host: str = "http://host.docker.internal:11434"
    model_name: str = "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0"
    
    # Konfiguracja retrieval i rerankingu
    retrieval_top_k: int = 100         # Ile chunków pobrać retrieval modelem
    rerank_top_k: int = 10             # Ile chunków po rerankingu do kontekstu
    
    # Pozostałe ustawienia
    max_context_length: int = 4000
    use_reranking: bool = True


class ElasticsearchRetrievalClient:
    """Klient do wyszukiwania semantycznego z wykorzystaniem embeddingów"""
    
    def __init__(self, host: str, index_name: str, embedding_service_url: str):
        self.es = Elasticsearch([host])
        self.index_name = index_name
        self.embedding_service_url = embedding_service_url
        logger.info(f"Używam serwisu embeddingów pod: {embedding_service_url}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Wywołuje serwis embeddingów via REST API"""
        try:
            response = requests.post(
                f"{self.embedding_service_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if embedding is None:
                raise ValueError("Brak embeddingu w odpowiedzi serwisu")
            return embedding
        except Exception as e:
            logger.error(f"Błąd wywołania serwisu embeddingów: {e}")
            raise
    
    def search_with_embeddings(self, query: str, top_k: int = 50) -> List[DocumentChunk]:
        """
        Wyszukiwanie semantyczne używając embeddingów
        """
        try:
            # Generuj embedding dla zapytania przez serwis REST
            logger.info(f"Generowanie embeddingu dla zapytania przez serwis embeddingów...")
            query_embedding = self._get_embedding(query)
            
            # Wyszukiwanie kNN w Elasticsearch
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
                "_source": ["text"]
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            chunks = []
            for hit in response['hits']['hits']:
                chunk = DocumentChunk(
                    content=hit['_source'].get('text', ''),
                    retrieval_score=hit['_score'],
                    chunk_id=hit['_id']
                )
                chunks.append(chunk)
            
            logger.info(f"✅ Retrieval zwrócił {len(chunks)} chunków (top-{top_k})")
            return chunks
            
        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania: {e}")
            return []


class PolishReranker:
    """Reranker dla języka polskiego używający sdadas/polish-reranker-roberta-v3"""
    
    def __init__(self):
        logger.info("Ładowanie modelu rerankingowego: sdadas/polish-reranker-roberta-v3...")
        
        self.model_name = 'sdadas/polish-reranker-roberta-v3'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Ustaw na tryb ewaluacji
        self.model.eval()
        
        # Sprawdź czy GPU jest dostępne
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"✅ Model rerankingowy załadowany na: {self.device}")
    
    def rerank_chunks(self, query: str, chunks: List[DocumentChunk], top_k: int = 6) -> List[DocumentChunk]:
        """
        Reranking chunków używając polish-reranker-roberta-v3
        """
        if not chunks:
            return chunks
        
        logger.info(f"🎯 RERANKING: {len(chunks)} chunków → wybór top {top_k}")
        
        try:
            # Przygotuj pary query-document
            pairs = [[query, chunk.content] for chunk in chunks]
            
            # Batch processing dla wydajności
            batch_size = 16
            all_scores = []
            
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i + batch_size]
                    
                    # Tokenizacja
                    inputs = self.tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Predykcja
                    outputs = self.model(**inputs)
                    scores = outputs.logits.squeeze(-1).cpu().numpy()
                    
                    # Jeśli pojedynczy wynik, przekonwertuj na listę
                    if scores.ndim == 0:
                        scores = [float(scores)]
                    else:
                        scores = scores.tolist()
                    
                    all_scores.extend(scores)
            
            # Przypisz rerank scores
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(all_scores[i])
            
            # Sortuj według rerank score (wyższy = lepszy)
            reranked_chunks = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
            
            # Logowanie top scores
            logger.info("📊 Top 3 rerank scores:")
            for i, chunk in enumerate(reranked_chunks[:3]):
                preview = chunk.content[:100].replace('\n', ' ')
                logger.info(f"  {i+1}. Score: {chunk.rerank_score:.4f} | Preview: {preview}...")
            
            return reranked_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Błąd podczas rerankingu: {e}")
            # Fallback - zwróć oryginalne chunki
            return chunks[:top_k]


class OllamaClient:
    """Klient do komunikacji z Ollama"""
    
    def __init__(self, host: str, model_name: str):
        self.host = host
        self.model_name = model_name
        
    def generate_response(self, context: str, question: str) -> str:
        """Generuje odpowiedź na podstawie kontekstu"""
        
        prompt = self._create_legal_prompt(context, question)
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                timeout=1000
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Brak odpowiedzi z modelu')
            else:
                logger.error(f"Błąd HTTP {response.status_code}: {response.text}")
                return "Nie udało się uzyskać odpowiedzi z modelu"
                
        except requests.exceptions.Timeout:
            logger.error("Timeout podczas komunikacji z Ollama")
            return "Przekroczono czas oczekiwania na odpowiedź"
        except Exception as e:
            logger.error(f"Błąd podczas generowania odpowiedzi: {e}")
            return f"Wystąpił błąd: {str(e)}"
    
    def _create_legal_prompt(self, context: str, question: str) -> str:
        """Prompt dla asystenta prawnego"""
        
        prompt = f"""Jesteś asystentem prawnym specjalizującym się w polskim prawie. Na podstawie podanych fragmentów dokumentów odpowiedz na pytanie użytkownika.

NAJLEPSZE FRAGMENTY DOKUMENTÓW (wybrane przez system retrieval + reranking):
{context}

PYTANIE UŻYTKOWNIKA:
{question}

INSTRUKCJE:
1. Odpowiadaj wyłącznie na podstawie podanych fragmentów
2. Fragmenty zostały wybrane jako NAJLEPSZE spośród wielu opcji
3. Jeśli pytanie dotyczy definicji, zacytuj dokładnie odpowiedni fragment
4. Używaj jasnego, zrozumiałego języka
5. Jeśli nie ma wystarczających informacji w podanych fragmentach, napisz to jasno
6. Nie dodawaj informacji spoza podanych fragmentów

ODPOWIEDŹ:"""
        
        return prompt


class LegalRAGSystem:
    """Główna klasa systemu RAG z dwuetapowym przetwarzaniem"""
    
    def __init__(self, config: LegalRAGConfig):
        self.config = config
        
        # Inicjalizacja komponentów
        self.es_client = ElasticsearchRetrievalClient(
            config.elasticsearch_host, 
            config.elasticsearch_index,
            config.embedding_service_url
        )
        
        if config.use_reranking:
            self.reranker = PolishReranker()
        else:
            self.reranker = None
            
        self.ollama_client = OllamaClient(
            config.ollama_host, 
            config.model_name
        )
    
    def process_question(self, question: str) -> dict:
        """
        Dwuetapowy pipeline:
        1. Retrieval: sdadas/mmlw-retrieval-roberta-large → top-50
        2. Reranking: sdadas/polish-reranker-roberta-v3 → top-6
        3. Generowanie odpowiedzi przez PLLuM
        """
        
        logger.info(f"🔍 Przetwarzanie pytania: {question}")
        
        # ETAP 1: RETRIEVAL - pobierz top-K kandydatów
        logger.info(f"📊 ETAP 1: Retrieval (top-{self.config.retrieval_top_k})...")
        retrieved_chunks = self.es_client.search_with_embeddings(
            question, 
            self.config.retrieval_top_k
        )
        
        if not retrieved_chunks:
            return {
                'answer': 'Nie znaleziono odpowiednich dokumentów dla Twojego pytania.',
                'chunks': [],
                'confidence': 0.0,
                'pipeline_stats': {
                    'retrieved': 0,
                    'reranked': 0,
                    'final': 0,
                    'reranking_used': False
                }
            }
        
        # ETAP 2: RERANKING - wybierz najlepsze fragmenty
        if self.config.use_reranking and self.reranker:
            logger.info(f"🎯 ETAP 2: Reranking (top-{self.config.rerank_top_k})...")
            final_chunks = self.reranker.rerank_chunks(
                question, 
                retrieved_chunks,
                self.config.rerank_top_k
            )
            reranking_used = True
        else:
            final_chunks = retrieved_chunks[:self.config.rerank_top_k]
            reranking_used = False
        
        # ETAP 3: Przygotowanie kontekstu
        context = self._prepare_context(final_chunks)
        
        # ETAP 4: Generowanie odpowiedzi
        logger.info("🧠 ETAP 3: Generowanie odpowiedzi przez LLM...")
        answer = self.ollama_client.generate_response(context, question)
        
        # Przygotowanie wyników
        confidence = self._calculate_confidence(final_chunks)
        
        pipeline_stats = {
            'retrieved': len(retrieved_chunks),
            'reranked': len(retrieved_chunks) if reranking_used else 0,
            'final': len(final_chunks),
            'reranking_used': reranking_used
        }
        
        return {
            'answer': answer,
            'chunks': final_chunks,
            'confidence': confidence,
            'pipeline_stats': pipeline_stats
        }
    
    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Przygotowuje kontekst z najlepszych chunków"""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            score_info = ""
            if chunk.rerank_score is not None:
                score_info = f" (rerank: {chunk.rerank_score:.3f})"
            else:
                score_info = f" (retrieval: {chunk.retrieval_score:.3f})"
            
            fragment = f"""
FRAGMENT {i}{score_info}:
{chunk.content}
---"""
            
            if current_length + len(fragment) > self.config.max_context_length:
                logger.info(f"Osiągnięto limit kontekstu, używam {i-1} fragmentów")
                break
                
            context_parts.append(fragment)
            current_length += len(fragment)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, chunks: List[DocumentChunk]) -> float:
        """Oblicza pewność na podstawie scores"""
        
        if not chunks:
            return 0.0
        
        # Użyj rerank scores jeśli dostępne
        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_scores:
            # Normalizacja sigmoid-like
            max_score = max(rerank_scores)
            min_score = min(rerank_scores)
            
            if max_score > min_score:
                normalized = [(score - min_score) / (max_score - min_score) for score in rerank_scores]
                return sum(normalized) / len(normalized)
            else:
                return 0.5
        
        # Fallback do retrieval scores
        retrieval_scores = [c.retrieval_score for c in chunks]
        if retrieval_scores:
            max_score = max(retrieval_scores)
            if max_score > 0:
                normalized = [score / max_score for score in retrieval_scores]
                return sum(normalized) / len(normalized)
        
        return 0.0


def main():
    """Główna funkcja aplikacji"""
    
    print("🏛️  System RAG z Polskim Rerankingiem")
    print("=" * 70)
    print("📊 Pipeline dwuetapowy:")
    print("   1️⃣  RETRIEVAL: sdadas/mmlw-retrieval-roberta-large → top-50")
    print("   2️⃣  RERANKING: sdadas/polish-reranker-roberta-v3 → top-6")
    print("   3️⃣  GENERATION: PLLuM (Ollama)")
    print("=" * 70)
    print("Wpisz 'exit' aby zakończyć\n")
    
    # Konfiguracja systemu
    config = LegalRAGConfig()
    
    # Możliwość override z zmiennych środowiskowych
    config.elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', config.elasticsearch_host)
    config.elasticsearch_index = os.getenv('ELASTICSEARCH_INDEX', config.elasticsearch_index)
    config.embedding_service_url = os.getenv('EMBEDDING_SERVICE_URL', config.embedding_service_url)
    config.ollama_host = os.getenv('OLLAMA_HOST', config.ollama_host)
    config.use_reranking = os.getenv('USE_RERANKING', 'true').lower() == 'true'
    config.retrieval_top_k = int(os.getenv('RETRIEVAL_TOP_K', '50'))
    config.rerank_top_k = int(os.getenv('RERANK_TOP_K', '6'))
    
    try:
        # Inicjalizacja systemu
        logger.info("Inicjalizacja systemu RAG...")
        rag_system = LegalRAGSystem(config)
        
        print("\n✅ System zainicjalizowany!")
        print(f"📊 Konfiguracja:")
        print(f"   • Retrieval top-K: {config.retrieval_top_k}")
        print(f"   • Reranking top-K: {config.rerank_top_k}")
        print(f"   • Reranking: {'✅ WŁĄCZONY' if config.use_reranking else '❌ WYŁĄCZONY'}")
        print(f"   • Max długość kontekstu: {config.max_context_length}")
        print()
        
        while True:
            question = input("💬 Twoje pytanie: ").strip()
            
            if question.lower() in ['exit', 'quit', 'koniec']:
                print("👋 Do widzenia!")
                break
                
            if not question:
                continue
            
            print("\n" + "="*70)
            print("🔄 PIPELINE W AKCJI:")
            print("="*70)
            
            # Przetwarzanie pytania
            result = rag_system.process_question(question)
            
            # Wyświetlenie odpowiedzi
            print(f"\n📖 ODPOWIEDŹ:")
            print("-" * 70)
            print(result['answer'])
            
            # Statystyki pipeline'u
            stats = result['pipeline_stats']
            print(f"\n📊 STATYSTYKI PIPELINE'U:")
            print(f"   1️⃣  Retrieval: {stats['retrieved']} chunków")
            if stats['reranking_used']:
                print(f"   2️⃣  Reranking: {stats['reranked']} → {stats['final']} chunków")
            else:
                print(f"   2️⃣  Reranking: POMINIĘTY")
            print(f"   3️⃣  Finalne chunki w kontekście: {stats['final']}")
            print(f"   📈 Poziom pewności: {result['confidence']:.2%}")
            
            # Wyświetlenie najlepszych chunków
            if result['chunks']:
                print(f"\n📚 NAJLEPSZE FRAGMENTY:")
                for i, chunk in enumerate(result['chunks'], 1):
                    print(f"\n{i}. {'🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '📄'}")
                    if chunk.rerank_score is not None:
                        print(f"   🎯 Rerank Score: {chunk.rerank_score:.4f}")
                    print(f"   📊 Retrieval Score: {chunk.retrieval_score:.3f}")
                    preview = chunk.content.replace('\n', ' ')
                    idxx = chunk.chunk_id
                    print(f"   📝 Fragment: {preview}...{idxx}")
            
            print("\n" + "="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n👋 Do widzenia!")
    except Exception as e:
        logger.error(f"Błąd krytyczny: {e}", exc_info=True)
        print(f"❌ Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
