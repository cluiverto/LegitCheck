import time, re, os
import requests
from urllib.parse import urlparse
from datetime import datetime
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

def generate_filename_from_content(soup, url):
    """
    Generuje nazwę pliku na podstawie zawartości strony
   
    Args:
        soup: Obiekt BeautifulSoup
        url: URL strony
   
    Returns:
        str: Nazwa pliku bez rozszerzenia
    """
   
    # Próbuj pobrać tytuł strony
    title = None
   
    # Najpierw sprawdź tag title
    title_tag = soup.find('title')
    if title_tag and title_tag.get_text().strip():
        title = title_tag.get_text().strip()
   
    # Jeśli nie ma title, sprawdź h1
    if not title:
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.get_text().strip():
            title = h1_tag.get_text().strip()
   
    # Jeśli nie ma h1, sprawdź h2
    if not title:
        h2_tag = soup.find('h2')
        if h2_tag and h2_tag.get_text().strip():
            title = h2_tag.get_text().strip()
   
    # Jeśli znaleziono tytuł, użyj go
    if title:
        # Oczyść tytuł
        filename = re.sub(r'[^\w\s\-_]', '', title)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename[:50]  # Ogranicz długość
        return filename
   
    # Jeśli nie ma tytułu, użyj URL-a z większą szczegółowością
    parsed_url = urlparse(url)
   
    # Użyj części ścieżki
    path_parts = [part for part in parsed_url.path.split('/') if part]
   
    if path_parts:
        # Weź ostatnie 2-3 części ścieżki
        filename_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
        filename = '_'.join(filename_parts)
       
        # Dodaj parametry query jeśli istnieją
        if parsed_url.query:
            query_clean = re.sub(r'[^\w\-_]', '_', parsed_url.query)[:20]
            filename += f"_{query_clean}"
    else:
        # Jeśli nie ma ścieżki, użyj domeny
        filename = parsed_url.netloc.replace('.', '_')
   
    # Oczyść nazwę pliku
    filename = re.sub(r'[^\w\-_]', '_', filename)
    filename = re.sub(r'_+', '_', filename)  # Usuń wielokrotne podkreślenia
    filename = filename.strip('_')
   
    return filename


def save_html_from_url(url, folder_path="html_files", filename=None):
    """
    Pobiera HTML z URL i zapisuje do pliku z unikalną nazwą
   
    Args:
        url (str): URL do pobrania
        folder_path (str): Folder do zapisania pliku
        filename (str): Nazwa pliku (opcjonalne, domyślnie z zawartości/URL)
   
    Returns:
        str: Ścieżka do zapisanego pliku lub None jeśli błąd
    """
   
    try:
        # Utwórz folder jeśli nie istnieje
        os.makedirs(folder_path, exist_ok=True)
       
        # Pobierz stronę
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
       
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
       
        # Parsuj HTML
        soup = BeautifulSoup(response.content, 'html.parser')
       
        # Generuj nazwę pliku jeśli nie podano
        if not filename:
            filename = generate_filename_from_content(soup, url)
        else:
            # Usuń rozszerzenie jeśli jest
            filename = re.sub(r'\.[^.]*$', '', filename)
       
        # Oczyść nazwę pliku
        filename = re.sub(r'[^\w\-_]', '_', filename)
       
        # Sprawdź czy plik już istnieje i dodaj timestamp jeśli tak
        base_filename = filename
        counter = 1
        while True:
            full_filename = f"{filename}.html"
            file_path = os.path.join(folder_path, full_filename)
           
            if not os.path.exists(file_path):
                break
           
            # Dodaj licznik lub timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_filename}_{timestamp}"
            counter += 1
       
        # Zapisz HTML
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
       
        print(f"Zapisano: {file_path}")
        return file_path
       
    except requests.RequestException as e:
        print(f"Błąd pobierania {url}: {e}")
        return None
    except Exception as e:
        print(f"Błąd zapisywania {url}: {e}")
        return None


def save_multiple_urls(urls, folder_path="html_files", delay=1):
    """
    Pobiera i zapisuje wiele URL-i z unikalnymi nazwami
   
    Args:
        urls (list): Lista URL-i
        folder_path (str): Folder do zapisania
        delay (int): Opóźnienie między pobieraniami (sekundy)
   
    Returns:
        list: Lista ścieżek do zapisanych plików
    """
   
    saved_files = []
   
    for i, url in enumerate(urls):
        print(f"Pobieranie {i+1}/{len(urls)}: {url}")
       
        file_path = save_html_from_url(url, folder_path)
        if file_path:
            saved_files.append(file_path)
       
        # Opóźnienie między pobieraniami
        if i < len(urls) - 1:
            time.sleep(delay)
   
    print(f"\nPobrano {len(saved_files)} z {len(urls)} plików")
    return saved_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Konwertuj pliki HTML na TXT z nazwą z <h1>.")
    parser.add_argument("input_folder", help="Ścieżka do folderu z plikami HTML")
    parser.add_argument("output_folder", help="Ścieżka do folderu, gdzie zostaną zapisane pliki TXT")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
