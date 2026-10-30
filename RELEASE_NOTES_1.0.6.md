# LinguaBee 1.0.6

## Krótki opis (do GitHub Release)

LinguaBee 1.0.6 pozwala wybrać kartę AMD w ustawieniach programu: lista urządzeń w ustawieniach Whisper, NLLB i streszczeń pokazuje wszystkie karty AMD, a wybór jest zapamiętywany i stosowany przy następnym uruchomieniu.

## Najważniejsze zmiany

### Wybór karty AMD w GUI

- **Wszystkie karty AMD na liście urządzeń:** w 1.0.5 program widział tylko kartę ustawioną w `HIP_VISIBLE_DEVICES` (domyślnie pierwszą), więc np. zewnętrzny Radeon AI PRO R9700 nie pojawiał się w ustawieniach. Teraz lista zawiera wszystkie karty AMD (nazwy odczytywane w osobnym procesie), a nieaktywna jest oznaczona „– po ponownym uruchomieniu”.
- **Jedna karta na proces:** przy kilku kartach widocznych naraz PyTorch z ROCm wywraca się przy obliczeniach (0xC0000005, sprawdzone na 8060S + R9700, PyTorch 2.12 / ROCm 7.15). Dlatego wybór innej karty zapisuje ustawienie `amd_gpu` i wyświetla komunikat, że zmiana zadziała po ponownym uruchomieniu; dotyczy Whisper, NLLB i streszczeń.
- **`main.py`:** przed importem torch ustawia `HIP_VISIBLE_DEVICES` z ustawienia `amd_gpu` (domyślnie 0). Ręcznie podana zmienna `HIP_VISIBLE_DEVICES` ma pierwszeństwo.

## Zmienione pliki

- `main.py`
- `gui/main_window.py`
- `README.md`
- `packaging/LinguaBee-macos.spec` (APP_VERSION 1.0.6)
- `setup.iss` (AppVersion 1.0.6)
- `RELEASE_NOTES_1.0.6.md` (nowy)

## Tag

- Tag: `v1.0.6`
