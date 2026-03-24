# ⚖️ LegitCheck

Interaktywna sandbox aplikacji prawnej z wykorzystaniem Retrieval‑Augmented Generation (RAG), Elasticsearch i local‑LLMów z Ollamy.

---

## 🔧 Architektura

System składa się z kilku kluczowych komponentów:

- **Backend**: Python, `sentence-transformers` + `transformers`
  - Retrieval embeddings: `sdadas/mmlw-retrieval-roberta-large`
  - Reranking: `sdadas/polish-reranker-roberta-v3`
- **Baza / search**:
  - `Elasticsearch` – przechowywanie i wyszukiwanie prawnych dokumentów.
- **LLM / generacja**:
  - `Ollama` – obsługa modeli typu `SpeakLeash/bielik-*` oraz `PRIHLOP/PLLuM`.
- **Frontend**:
  - `Streamlit` – użytkowski interfejs z:
    - wyszukiwarką fraz,
    - sekcją pytań z wyświetleniem źródeł i statystyk.

---

## 📦 Funkcje

- **Zadawanie pytań prawnych** w języku polskim.
- **Dokumentacja odpowiedzi**: zaznaczenie z jakich fragmentów (artykuł, paragraf, id) została wzięta informacja.
- **Reranking**: zwiększenie jakości odpowiedzi poprzez reranking najbardziej istotnych fragmentów z `polish‑reranker‑roberta‑v3`.
- **Wyszukiwarka fraz**:
  - Wpiszesz dowolną frazę → system policzy, ile razy występuje w dokumentach zindeksowanych w Elasticsearch.
- **Panel administracyjny**:
  - Wybór modelu LLM (np. Bielik 4.5 / 1.5, PLLuM).
  - Ustawienie `retrieval top‑K` i `rerank top‑K`.
  - Reset historii rozmów.

---

## 🚀 Szybki start

### 1. Wymagania

- Python 3.10+
- Elasticsearch podany jako `ELASTICSEARCH_HOST` (z przygotowanym indeksem).
- Ollama uruchomiona pod `OLLAMA_HOST`.

