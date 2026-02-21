# LinguaBee 0.9.4

## Krótki opis (do GitHub Release)

LinguaBee 0.9.4 wprowadza nowy panel **Wynik pracy** z podglądem rezultatów transkrypcji, tłumaczenia i streszczenia, poprawia ergonomię layoutu oraz uniezależnia podgląd od wyboru formatów zapisu plików.

## Najważniejsze zmiany

- Dodano sekcję **Wynik pracy** po prawej stronie GUI z przełączanymi przyciskami:
  - Transkrypcja,
  - Tłumaczenie,
  - Streszczenie.
- Dodano przycisk **Kopiuj** dla aktualnie wybranego podglądu.
- Podgląd jest aktualizowany na podstawie sygnałów z workera i działa dla wielu plików.
- Ujednolicono nazwy modeli w UI:
  - `Gemini` (zamiast `Gemini (API)`),
  - `OpenRouter` (zamiast `OpenRouter (API)`).
- Dodano zgodność wsteczną z zapisanymi ustawieniami zawierającymi stare etykiety modeli.
- Usprawniono layout:
  - zamrożono szerokości sekcji po lewej,
  - sekcja **Wynik pracy** rozszerza się przy poszerzaniu okna,
  - zwiększono domyślną szerokość całego GUI.
- Zmieniono styl pola wyników:
  - tło: ciemnoszare,
  - tekst: biały,
  - większa waga fontu.
- Podgląd wyników jest **niezależny od formatów wyjściowych**:
  - Transkrypcja: zawsze podgląd w **SRT**,
  - Tłumaczenie: zawsze podgląd w **SRT**,
  - Streszczenie: zawsze podgląd w **TXT**,
  - brak zaznaczonych formatów nie blokuje działania modeli ani podglądu.

## Zmienione pliki

- `gui/main_window.py`
- `gui/widgets.py`
- `gui/stylesheet.qss`
- `gui/dialogs.py`
- `worker.py`
- `translation_manager.py`
- `summarization_manager.py`
- `setup.iss`
- `RELEASE_NOTES_0.9.4.md`

## Tag i commit

- Tag: `v0.9.4`
