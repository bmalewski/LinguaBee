@echo off
rem Instalacja LinguaBee ze zrodel: srodowisko Pythona (venv), PyTorch dobrany do karty,
rem pozostale biblioteki i skrot na pulpicie. Wymaga Pythona 3.12 i Gita (patrz README).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
echo.
pause
