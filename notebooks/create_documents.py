import os
import re
from typing import List, Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class CustomDocument:
    text: str
    metadata: dict
    doc_id: str

def extract_title_and_create_document(html_file_path: str) -> Optional[CustomDocument]:
    """
    Wyciąga tytuł z h1 i tworzy niestandardowy obiekt dokumentu
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        h1_tag = soup.find('h1')
        if h1_tag:
            for sup in h1_tag.find_all("sup"):
                sup.decompose()
            for a in h1_tag.find_all("a", class_="gloss-link tooltip"):
                a.decompose()

        title = h1_tag.get_text(strip=True) if h1_tag else "Brak tytułu"

        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        metadata = {
            'title': title,
            'source_file': os.path.basename(html_file_path),
            'source_path': html_file_path,
            'text_length': len(text),
            'doc_type': 'ustawa'
        }

        document = CustomDocument(
            text=text,
            metadata=metadata,
            doc_id=os.path.splitext(os.path.basename(html_file_path))[0]
        )

        return document
    except Exception as e:
        print(f"Błąd przetwarzania {html_file_path}: {e}")
        return None

def process_html_folder(folder_path: str) -> List[CustomDocument]:
    """
    Przetwarza wszystkie pliki HTML w folderze
    """
    documents = []

    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]

    print(f"Znaleziono {len(html_files)} plików HTML")

    for html_file in html_files:
        file_path = os.path.join(folder_path, html_file)
        print(f"Przetwarzanie: {html_file}")

        document = extract_title_and_create_document(file_path)
        if document:
            documents.append(document)
            print(f"  ✓ Tytuł: {document.metadata['title'][:60]}...")
        else:
            print(f"  ✗ Błąd przetwarzania")

    print(f"\nUtworzono {len(documents)} dokumentów")
    return documents

def print_documents_info(documents: List[CustomDocument]):
    """
    Wyświetla informacje o dokumentach
    """
    print("\n=== INFORMACJE O DOKUMENTACH ===")
    for i, doc in enumerate(documents, 1):
        print(f"\n{i}. {doc.metadata['title']}")
        print(f"   ID: {doc.doc_id}")
        print(f"   Plik: {doc.metadata['source_file']}")
        print(f"   Długość: {doc.metadata['text_length']} znaków")
        print(f"   Tekst: {doc.text[:100]}...")

if __name__ == "__main__":
    documents = process_html_folder("ustawy")
    print_documents_info(documents)
    print(f"\nGotowe! Masz {len(documents)} dokumentów do dalszego przetwarzania.")
