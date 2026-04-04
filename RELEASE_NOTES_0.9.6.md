# LinguaBee 0.9.6

## Krótki opis (do GitHub Release)

LinguaBee 0.9.6 wprowadza nowe ustawienia formatowania napisów SRT bezpośrednio w oknie Whisper. Możesz teraz precyzyjnie kontrolować liczbę linii i maksymalną długość linii, co ułatwia dopasowanie napisów do preferencji czytelności i standardów publikacji.

## Najważniejsze zmiany

- Dodano nowe opcje w oknie ustawień Whisper:
  - **Maksymalna liczba linii w napisach (SRT)**
  - **Maksymalna liczba znaków w linii SRT**
- Ustawiono domyślne wartości:
  - **2 linie**
  - **25 znaków**
- Nowe ustawienia są zapisywane i odczytywane z konfiguracji użytkownika.
- Parametry formatowania SRT są przekazywane do pipeline'u i używane podczas zapisu plików transkrypcji.
- Ulepszono zapis SRT:
  - automatyczne zawijanie tekstu do limitu znaków,
  - ograniczenie liczby linii w pojedynczym cue,
  - dzielenie dłuższych segmentów na kolejne cue z proporcjonalnym podziałem czasu.

## Zmienione pliki

- `gui/dialogs.py`
- `gui/main_window.py`
- `config.py`
- `artifact_io_service.py`
- `file_utils.py`
- `worker.py`

## Tag i commit

- Tag: `v0.9.6`
- Commit: _uzupełni się po utworzeniu wydania_
