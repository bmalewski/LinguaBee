@echo off
rem Tworzy skrot "LinguaBee" (z ikona) na pulpicie i w menu Start.
rem Skrot uruchamia program bezposrednio, bez migajacego okna konsoli.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\windows\create_shortcut.ps1" -Root "%~dp0."
echo.
pause
