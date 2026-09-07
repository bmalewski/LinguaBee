# LinguaBee 1.0.3

## Krótki opis (do GitHub Release)

LinguaBee 1.0.3 to drobne wydanie wizualne: logo w oknie programu ma przezroczyste tło i zlewa się z tłem interfejsu, zamiast wyświetlać się jako kremowy prostokąt.

## Najważniejsze zmiany

- **Logo z przezroczystym tłem:** nowy plik `icons/LinguaBee_logo_transparent.png` (pszczoła i napis bez tła, miękkie krawędzie, przezroczyste otwory w literach, przycięty do zawartości). `gui/main_window.py` używa go dla logo w oknie i ikony okna. Dotychczasowy `icons/LinguaBee_1.0_512.png` pozostaje w repozytorium.
- `packaging/LinguaBee-macos.spec`: nowy plik logo dołączany do aplikacji macOS.

## Zmienione pliki

- `icons/LinguaBee_logo_transparent.png` (nowy)
- `gui/main_window.py`
- `packaging/LinguaBee-macos.spec` (APP_VERSION 1.0.3)
- `setup.iss` (AppVersion 1.0.3)
- `RELEASE_NOTES_1.0.3.md` (nowy)

## Tag

- Tag: `v1.0.3`
