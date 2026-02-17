import os
import re
from bs4 import BeautifulSoup

def extract_text_and_save_as_txt(html_file_path: str, output_folder: str) -> None:
    """
    Wyciąga cały tekst z pliku HTML i zapisuje go do pliku .txt
    w folderze output_folder. Nazwa pliku to tekst z <h1>.
    
    Args:
        html_file_path (str): Ścieżka do pliku HTML
        output_folder (str): Ścieżka do folderu, gdzie zapisać plik .txt
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Wyciągnij tytuł z <h1>
        h1_tag = soup.find('h1')
        if h1_tag:
            # Usuń znaczniki <sup> i linki klasy "gloss-link tooltip" wewnątrz h1
            for sup in h1_tag.find_all("sup"):
                sup.decompose()
            for a in h1_tag.find_all("a", class_="gloss-link tooltip"):
                a.decompose()
            title = h1_tag.get_text(strip=True)
        else:
            title = "brak_tytulu"

        # Oczyść nazwę pliku - usuń niedozwolone znaki w nazwach plików
        safe_title = re.sub(r'[\\/*?:"<>|]', '_', title)

        # Pobierz cały tekst z pliku (bez HTML)
        text = soup.get_text(separator='\n', strip=True)

        # Oczyść tekst z nadmiernych białych znaków
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        # Upewnij się, że folder wyjściowy istnieje
        os.makedirs(output_folder, exist_ok=True)

        # Ścieżka do pliku txt
        output_path = os.path.join(output_folder, f"{safe_title}.txt")

        # Zapisz tekst do pliku .txt
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(text)

        print(f"Zapisano plik: {output_path}")

    except Exception as e:
        print(f"Błąd przetwarzania pliku {html_file_path}: {e}")

def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Przetwarza wszystkie pliki HTML w folderze input_folder,
    wyciąga tekst i zapisuje do plików .txt w output_folder.
    """
    if not os.path.isdir(input_folder):
        print(f"Folder {input_folder} nie istnieje.")
        return
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.html', '.htm')):
            path = os.path.join(input_folder, filename)
            extract_text_and_save_as_txt(path, output_folder)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Konwertuj pliki HTML na TXT z nazwą z <h1>.")
    parser.add_argument("input_folder", help="Ścieżka do folderu z plikami HTML")
    parser.add_argument("output_folder", help="Ścieżka do folderu, gdzie zostaną zapisane pliki TXT")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
