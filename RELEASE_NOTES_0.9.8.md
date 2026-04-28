# LinguaBee 0.9.8

## Co nowego?

LinguaBee 0.9.8 skupia się na poprawie stabilności modułu transkrypcji Whisper, rozwiązując uciążliwy problem halucynacji i pomijania partii nagrania.

### Zmiany i ulepszenia
- **Wbudowano Voice Activity Detection (VAD):** Uruchomienie `vad_filter=True` znacząco redukuje do zera halucynacje i uciążliwe transkrypcje szumów.
- **Odblokowano fallback mode:** Usunięto ucięte na sztywno ograniczenia temperaturowe (`temperature=0`) oraz blokady powtórzeń (`repetition_penalty`, `no_repeat_ngram_size`), co w naturalny sposób przywraca domyślną odporność modelu Whisper na braki i pomijanie trudniejszych technicznie fragmentów nagrania.
- **Optymalizacja beam_size:** Przywrócono sugerowaną wartość `beam_size=5` (zamiast 10), odciążając czas pracy przy zachowaniu niemal identycznej jakości detekcji słów.
- **Modyfikacja promptowania kontekstu:** Zmieniono na `condition_on_previous_text=False`, przez co błędy wcześniejszych fragmentów nie wciągają modelu w zapętlone halucynacje.

## Zmiany techniczne
- `whisper_transcription.py` (Modyfikacja argumentów funkcji `.transcribe()`)
- `setup.iss` (AppVersion 0.9.8)

- Tag: `v0.9.8`
