\# Wizja komputerowa robota mobilnego – detekcja i pomiar drzew oraz kamieni



\## Instrukcja uruchomienia (Windows)

1\. Zainstaluj \*\*Python 3.12\*\* (z zaznaczoną opcją \*Add python.exe to PATH\*).

2\. Otwórz terminal w tym folderze i zainstaluj zależności:

&#x20;  `pip install -r requirements.txt`

&#x20;  \*(Jeśli masz Pythona 3.15, wpisz: `pip install ultralytics opencv-python numpy ftfy regex tqdm \&\& pip install --no-deps https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip`)\*

3\. Uruchom skrypt:

&#x20;  `python test\_folder.py`

4\. Wygenerowane obrazy oraz tabela Excel (`wyniki\_pomiarow.csv`) zapiszą się w folderze `wyniki/`.

