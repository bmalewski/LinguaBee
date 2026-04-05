# LinguaBee 0.9.7

## Krótki opis (do GitHub Release)

LinguaBee 0.9.7 poprawia jakość i przewidywalność przetwarzania audio w pipeline Whisper. Wersja wprowadza bezpieczniejszą obróbkę (łagodniejsze odszumianie i wyrównanie głośności), lepszą obsługę kanałów audio oraz szybkie presety jakości bezpośrednio w oknie ustawień.

## Najważniejsze zmiany

- Ulepszono tor przetwarzania audio:
  - łagodniejsze odszumianie,
  - łagodne wyrównanie głośności zamiast agresywnej normalizacji szczytowej,
  - zabezpieczenie przed clippingiem,
  - pomijanie odszumiania dla bardzo krótkich próbek.
- Poprawiono obsługę kanałów audio podczas przetwarzania (bardziej przewidywalne działanie dla stereo/mono).
- Zmieniono domyślne ustawienie mono na wyłączone (mniejsze ryzyko pogorszenia jakości przy standardowych nagraniach).
- Dodano szybkie presety jakości w oknie Whisper, w tym:
  - **Bez poprawy**,
  - **Lekka poprawa (wyrównanie)**,
  - **Szumy i głośność (odszumianie + wyrównanie)**,
  - oraz automatyczny tryb **Niestandardowe** po ręcznej zmianie checkboxów.
- Doprecyzowano opisy opcji audio w GUI, aby łatwiej dobrać właściwy tryb.

## Zmienione pliki

- `audio_processing.py`
- `config.py`
- `gui/dialogs.py`
- `gui/main_window.py`

## Tag i commit

- Tag: `v0.9.7`
- Commit: `e82b229`
