"""
FastAPI serwis embedingów z sentence-transformers
Obsługuje modele: sdadas/mmlw-retrieval-roberta-large i inne

Uruchomienie:
    pip install fastapi uvicorn sentence-transformers
    python embedding_service.py

Lub z dockerfile:
    docker build -t embedding-service .
    docker run -p 8000:8000 embedding-service
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from typing import List
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Embedding Service", version="1.0")


# Globalne zmienne
model = None
model_name = None


# Model embedingów
MODEL_NAME = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large")


def load_model():
    """Ładuje model z obsługą błędów huggingface_hub"""
    global model, model_name
    
    try:
        logger.info(f"Ładowanie modelu: {MODEL_NAME}...")
        
        # Importuj tutaj aby uniknąć konfliktów wersji
        from sentence_transformers import SentenceTransformer
        
        # Ustaw zmienne środowiskowe aby uniknąć problemów z cache
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
        
        # Załaduj model z timeout'em
        model = SentenceTransformer(MODEL_NAME)
        model_name = MODEL_NAME
        
        logger.info(f"✅ Model załadowany!")
        logger.info(f"   Wymiar embedingu: {model.get_sentence_embedding_dimension()}")
        
        return True
    
    except ImportError as e:
        if "cached_download" in str(e):
            logger.error(f"❌ Błąd importu huggingface_hub - problem z kompatybilnością wersji")
            logger.error(f"   Próbuję załadować model bez cached_download...")
            
            try:
                from sentence_transformers import SentenceTransformer
                # Wyłącz cache
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                model = SentenceTransformer(MODEL_NAME, cache_folder='/tmp/model_recache')
                model_name = MODEL_NAME
                logger.info(f"✅ Model załadowany (bez cache)")
                return True
            except Exception as e2:
                logger.error(f"❌ Nie udało się załadować modelu: {e2}")
                return False
        else:
            logger.error(f"❌ Błąd importu: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Błąd ładowania modelu: {e}")
        return False



class EmbedRequest(BaseModel):
    text: str



class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int



@app.on_event("startup")
async def startup_event():
    """Załaduj model przy starcie"""
    logger.info("🚀 Inicjalizacja serwisu...")
    if not load_model():
        logger.error("❌ Nie udało się załadować modelu - serwis będzie działać w trybie ograniczonym")



@app.get("/health")
async def health():
    """Health check"""
    if not model:
        return {
            "status": "loading",
            "model": MODEL_NAME,
            "message": "Model się ładuje..."
        }
    
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension(),
        "ready": True
    }



@app.get("/info")
async def info():
    """Informacje o serwisie"""
    if not model:
        return {
            "status": "error",
            "message": "Model nie załadowany",
            "model_name": MODEL_NAME
        }
    
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "status": "ready"
    }



@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generuje embedding dla tekstu"""
    
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model nie załadowany - czekaj lub spróbuj później"
        )
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Tekst nie może być pusty")
    
    try:
        text = request.text.strip()
        logger.info(f"Generowanie embeddingu dla tekstu ({len(text)} znaków)...")
        
        # Generuj embedding
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Konwertuj do listy
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)
        
        logger.info(f"✅ Embedding wygenerowany (wymiar: {len(embedding_list)})")
        
        return EmbedResponse(
            embedding=embedding_list,
            dimension=len(embedding_list)
        )
    except Exception as e:
        logger.error(f"❌ Błąd generowania embeddingu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd: {str(e)}")



@app.post("/embed_batch")
async def embed_batch(request: dict):
    """Generuje embedingi dla listy tekstów"""
    
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model nie załadowany"
        )
    
    texts = request.get("texts", [])
    
    if not texts:
        raise HTTPException(status_code=400, detail="Lista tekstów jest pusta")
    
    if not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="'texts' musi być listą")
    
    try:
        logger.info(f"Generowanie {len(texts)} embedingów...")
        
        # Oczyść teksty
        texts = [str(t).strip() for t in texts if str(t).strip()]
        
        if not texts:
            raise HTTPException(status_code=400, detail="Brak poprawnych tekstów")
        
        # Generuj embedingi
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Konwertuj do listy
        result_embeddings = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                result_embeddings.append(emb.tolist())
            else:
                result_embeddings.append(list(emb))
        
        logger.info(f"✅ {len(result_embeddings)} embedingów wygenerowanych")
        
        return {
            "embeddings": result_embeddings,
            "count": len(result_embeddings),
            "dimension": len(result_embeddings[0]) if result_embeddings else 0
        }
    except Exception as e:
        logger.error(f"❌ Błąd generowania batch embedingów: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd: {str(e)}")



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Embedding Service",
        "version": "1.0",
        "status": "ok" if model else "loading",
        "endpoints": {
            "/health": "Health check",
            "/info": "Informacje o serwisie",
            "/embed": "POST - Generuj embedding",
            "/embed_batch": "POST - Generuj batch embedingów"
        }
    }



if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Uruchamianie serwera na porcie {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1  # Jeden worker - mniej RAM
    )