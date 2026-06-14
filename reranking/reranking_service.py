"""
FastAPI serwis rerankingu z transformers
Obsługuje modele: sdadas/polish-reranker-roberta-v3

Uruchomienie:
    pip install -r requirements_reranking.txt
    python reranking_service.py

Lub z dockerfile:
    docker build -t reranking-service -f Dockerfile.reranking .
    docker run -p 8001:8001 reranking-service
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(title="Reranking Service", version="1.0")


# Globalne zmienne
model = None
tokenizer = None
device = None
model_name = None


# Model rerankingu
RERANK_MODEL = os.getenv("RERANK_MODEL", "sdadas/polish-reranker-roberta-v3")



def load_model():
    """Ładuje model rerankingu"""
    global model, tokenizer, device, model_name
    
    try:
        logger.info(f"Ładowanie modelu rerankingu: {RERANK_MODEL}...")
        
        # Ustaw zmienne środowiskowe dla cache
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info("Ładowanie tokenizera...")
        tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL,
            local_files_only=False,
            use_fast=True
        )
        logger.info("✅ Tokenizer załadowany")
        
        logger.info("Ładowanie modelu...")
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL,
            local_files_only=False,
            torch_dtype=torch.float32
        )
        
        # Ustaw na eval mode
        model.eval()
        
        # Sprawdź GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        model_name = RERANK_MODEL
        
        logger.info(f"✅ Model załadowany na: {device}")
        logger.info(f"   Model: {RERANK_MODEL}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Błąd ładowania modelu: {e}", exc_info=True)
        return False



class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 10



class RerankResponse(BaseModel):
    scores: List[float]
    ranked_indices: List[int]
    top_k: int



@app.on_event("startup")
async def startup_event():
    """Załaduj model przy starcie"""
    logger.info("🚀 Inicjalizacja serwisu rerankingu...")
    if not load_model():
        logger.error("❌ Nie udało się załadować modelu - serwis będzie działać w trybie ograniczonym")



@app.get("/health")
async def health():
    """Health check"""
    if not model:
        return {
            "status": "loading",
            "model": RERANK_MODEL,
            "message": "Model się ładuje..."
        }
    
    return {
        "status": "ok",
        "model": RERANK_MODEL,
        "device": str(device),
        "ready": True
    }



@app.get("/info")
async def info():
    """Informacje o serwisie"""
    if not model:
        return {
            "status": "error",
            "message": "Model nie załadowany",
            "model_name": RERANK_MODEL
        }
    
    return {
        "model_name": RERANK_MODEL,
        "device": str(device),
        "status": "ready"
    }



@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Reranking dokumentów"""
    
    if not model or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Model nie załadowany - czekaj lub spróbuj później"
        )
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query nie może być pusty")
    
    if not request.documents or len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="Lista dokumentów jest pusta")
    
    try:
        logger.info(f"Reranking {len(request.documents)} dokumentów...")
        
        # Przygotuj pary [query, document]
        pairs = [[request.query, doc] for doc in request.documents]
        
        logger.debug(f"Tokenizowanie {len(pairs)} par (max_length=512)...")
        
        # Tokenizuj z tymi samymi parametrami
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Przenieś na urządzenie
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.debug("Generowanie scores...")
        
        # Generuj scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Wyciągnij scores
        # Dla modeli z 2+ klasami (relevant/not-relevant) bierzemy score klasy pozytywnej
        if logits.dim() > 1 and logits.size(-1) > 1:
            scores = logits[:, 1]
        else:
            scores = logits.squeeze(-1)
        
        # Konwertuj na numpy a potem na listę
        if scores.dim() == 0:
            # Pojedynczy wynik
            scores_list = [float(scores.cpu().numpy())]
        else:
            scores_list = scores.cpu().numpy().tolist()
        
        logger.debug(f"Scores: {scores_list[:3]}...")
        
        # Posortuj i weź top-k
        ranked_indices = sorted(
            range(len(scores_list)), 
            key=lambda i: scores_list[i], 
            reverse=True
        )
        top_indices = ranked_indices[:request.top_k]
        top_scores = [scores_list[i] for i in top_indices]
        
        logger.info(f"✅ Reranking zakończony (top-{len(top_indices)})")
        logger.info(f"   Top 3 scores: {top_scores[:3]}")
        
        return RerankResponse(
            scores=top_scores,
            ranked_indices=top_indices,
            top_k=len(top_indices)
        )
    
    except Exception as e:
        logger.error(f"❌ Błąd rerankingu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd: {str(e)}")



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Reranking Service",
        "version": "1.0",
        "status": "ok" if model else "loading",
        "endpoints": {
            
            "/health": "Health check",
            "/info": "Informacje o serwisie",
            "/rerank": "POST - Rerank dokumenty"
        }
    }



if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    logger.info(f"Uruchamianie serwera rerankingu na porcie {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1
    )