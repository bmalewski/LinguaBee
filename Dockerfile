# 1. Wybór obrazu bazowego
FROM python:3.13-slim

# Ustawienie zmiennych środowiskowych dla Qt, aby działało w środowisku X11
ENV QT_QPA_PLATFORM=xcb

# 2. Instalacja zależności systemowych dla PySide6 (Qt6) i X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqt6gui6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    # Czyszczenie po instalacji w celu zmniejszenia rozmiaru obrazu
    && rm -rf /var/lib/apt/lists/*

# 3. Ustawienie katalogu roboczego
WORKDIR /app

# 4. Kopiowanie i instalacja zależności Python
# Kopiujemy najpierw ten plik, aby wykorzystać cache Dockera
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kopiowanie reszty kodu aplikacji
COPY . .

# 6. Uruchomienie aplikacji
CMD ["python", "main.py"]
