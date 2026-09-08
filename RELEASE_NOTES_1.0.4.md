# LinguaBee 1.0.4

## Krótki opis (do GitHub Release)

LinguaBee 1.0.4 rozdziela prompty korekty na osobne dla plików TXT/DOCX i dla napisów SRT, dodaje instalator macOS `.pkg` oraz instrukcję instalacji i odblokowania aplikacji na innym Macu.

## Najważniejsze zmiany

### Osobne prompty korekty (TXT/DOCX i SRT)

- **Dwa prompty zamiast jednego:** korekta napisów SRT wymaga instrukcji o zachowaniu segmentów i kodów czasowych, a korekta tekstu ciągłego (TXT/DOCX) nie. Dialogi korekty dla Ollama, Gemini i OpenRouter mają teraz dwie zakładki: „Prompt TXT / DOCX” i „Prompt SRT”, każda z własną listą szablonów (zapis/usuwanie).
- **Szablony:** prompty SRT korzystają z dotychczasowego katalogu `prompts/` (istniejące szablony użytkownika pozostają widoczne), prompty TXT/DOCX z nowego `prompts/correction_txt/`. Dodano szablony startowe `Korekta_SRT` i `Korekta_TXT_DOCX`.
- **`config.py`:** nowe pole `correction_prompt_srt`; `correction_prompt` dotyczy TXT/DOCX. Gdy prompt SRT jest pusty, `correction_service` używa promptu TXT/DOCX (zgodność wstecz ze starymi ustawieniami).
- **Walidacja przed startem** sprawdza tylko prompt potrzebny do zaznaczonych formatów oryginalnych i ostrzega, gdy korekta jest włączona bez żadnego formatu.
- **`gui/prompt_template_mixin.py`:** nowy widget `PromptTemplateEditor` (szablony + pole promptu) pozwalający umieścić kilka niezależnych edytorów w jednym dialogu. Dawny dialog korekty Ollama stał się klasą bazową `OllamaPromptSettingsDialog` dla dialogów streszczenia i tłumaczenia.

### Instalator macOS (.pkg)

- **`packaging/build_pkg.sh`:** buduje `dist/LinguaBee-<wersja>.pkg` z gotowego `dist/LinguaBee.app` narzędziami systemowymi (`pkgbuild`, `productbuild`). Instalator kopiuje aplikację do `/Applications` (bez relokacji), wymaga Apple Silicon i macOS 14+, ma polskie ekrany powitalny i końcowy (`packaging/pkg/*.html`) z instrukcją odblokowania.
- **`packaging/build_macos.sh --pkg`:** buduje aplikację i instalator w jednym kroku; opcje `--install` i `--pkg` można łączyć. Przy `--install` skrypt usuwa ewentualny znacznik kwarantanny.
- Instalator nie jest podpisany certyfikatem Apple: na innym Macu wymaga jednorazowego odblokowania (Ustawienia systemowe → Prywatność i ochrona → Otwórz mimo to).

### Dokumentacja

- README, sekcja 9: budowanie instalatora `.pkg`.
- README, sekcja 10: instrukcja dla odbiorcy — wymagania (Apple Silicon), odblokowanie instalatora i aplikacji w Gatekeeperze, alternatywa `xattr`, katalogi danych, najczęstsze pytania.

## Zmienione pliki

- `config.py`
- `correction_service.py`
- `gui/dialogs.py`
- `gui/main_window.py`
- `gui/prompt_template_mixin.py`
- `prompts/Korekta_SRT.txt` (nowy)
- `prompts/correction_txt/Korekta_TXT_DOCX.txt` (nowy)
- `packaging/build_pkg.sh` (nowy)
- `packaging/pkg/welcome.html`, `packaging/pkg/conclusion.html` (nowe)
- `packaging/build_macos.sh`
- `packaging/LinguaBee-macos.spec` (APP_VERSION 1.0.4)
- `README.md`
- `setup.iss` (AppVersion 1.0.4)
- `RELEASE_NOTES_1.0.4.md` (nowy)

## Tag

- Tag: `v1.0.4`
