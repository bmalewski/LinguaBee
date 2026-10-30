# LinguaBee 1.0.5

## Krótki opis (do GitHub Release)

LinguaBee 1.0.5 dodaje obsługę kart AMD przez ROCm: automatyczny instalator PyTorch dobierający build do sprzętu (NVIDIA / AMD / Apple / CPU), transkrypcję Whisper i tłumaczenie NLLB na karcie AMD (CTranslate2 z ROCm) oraz obsługę kilku kart AMD naraz (np. iGPU + zewnętrzna karta).

## Najważniejsze zmiany

### Instalator PyTorch dopasowany do sprzętu

- **Nowy `install_pytorch.py`:** wykrywa GPU i instaluje właściwy build PyTorch: NVIDIA → CUDA (cu128), AMD → ROCm z indeksu nightly AMD (`rocm.nightlies.amd.com/whl-multi-arch`), Apple Silicon → standardowy (MPS), brak GPU → CPU-only. Opcje: `--dry-run`, `--force cuda|rocm|cpu|mps`, `--gfx <target>`, `--list-targets`.
- **Kilka kart AMD:** przy np. Radeon 8060S + Radeon AI PRO R9700 instalowany jest kod dla wszystkich wykrytych targetów (`torch[device-gfx1151,device-gfx1201]`). Wcześniej tylko pierwsza karta miała swoje kernele i obliczenia na drugiej kończyły się błędem `hipErrorInvalidImage`.
- **`requirements.txt`** nie instaluje już torch bezpośrednio; `setup_windows.ps1`, `setup_macos.sh` i `Dockerfile` uruchamiają najpierw `install_pytorch.py`.

### Profil sprzętowy AMD GPU

- **`hardware_profile.py`:** nowy profil `amd_gpu` (PyTorch z ROCm rozpoznawany po `torch.version.hip`, próg pamięci 4 GB, bo iGPU korzysta z pamięci współdzielonej).
- **Whisper i NLLB na karcie AMD:** profil sprawdza, czy zainstalowany CTranslate2 widzi kartę (`get_cuda_device_count()`). Z paczką ROCm z wydań CTranslate2 na GitHubie (`rocm-python-wheels-*.zip`) oba silniki liczą na GPU; ze zwykłą paczką z PyPI zostają na CPU. Wcześniej profil AMD ustawiał NLLB na `cuda` bez względu na to, co skończyłoby się błędem przy CTranslate2 bez ROCm.
- **Whisper float16 na AMD:** `whisper_transcription.py` używa na ROCm `float16` zamiast `float32` (pomiar: ok. 2,4× szybciej, ten sam tekst).
- **Wybór karty:** `main.py` domyślnie ustawia `HIP_VISIBLE_DEVICES=0`, bo przy kilku widocznych kartach PyTorch z ROCm potrafi się wywrócić. Inną kartę wybiera się zmienną środowiskową przed uruchomieniem (np. `HIP_VISIBLE_DEVICES=1`). Na NVIDIA zmienna nie ma znaczenia.

### llama.cpp

- **`llama_cpp_translator.py`:** opis i komunikaty instalacji dla AMD (ROCm/HIP na Linuksie, Vulkan na Windows).

### Pomiary (Ryzen AI Max+ 395 / Radeon 8060S, Radeon AI PRO R9700 przez USB4, Windows 11)

faster-whisper large-v3, 5 min polskiej mowy, beam 5:

| Wariant | Czas |
|---|---|
| CPU int8 (beam 1) | ok. 260 s |
| 8060S float16 | 53–55 s |
| R9700 float16 | 48–55 s |
| Sam koder (80 okien 30 s): 8060S / R9700 | 19,6 s / 8,5 s |

Przy zwykłej transkrypcji obie karty są podobne, bo czas zjada dekoder (krok po kroku, opóźnienie wywołań), a nie moc karty.

## Dokumentacja

- README: sterowniki AMD, kroki instalacji z `install_pytorch.py`, Whisper/NLLB na karcie AMD (CTranslate2 z ROCm), wybór karty, rozwiązywanie problemów z ROCm.

## Zmienione pliki

- `install_pytorch.py` (nowy)
- `hardware_profile.py`
- `whisper_transcription.py`
- `main.py`
- `llama_cpp_translator.py`
- `requirements.txt`
- `setup_windows.ps1`, `setup_macos.sh`, `Dockerfile`
- `packaging/LinguaBee-macos.spec` (APP_VERSION 1.0.5)
- `setup.iss` (AppVersion 1.0.5)
- `README.md`
- `RELEASE_NOTES_1.0.5.md` (nowy)

## Tag

- Tag: `v1.0.5`
