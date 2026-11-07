@echo off
rem LinguaBee - uruchomienie dwuklikiem, bez okna konsoli.
rem Python jest szukany w: python\ (instalator), venv\ albo .venv\ (instalacja ze zrodel).
rem Gdy cos nie dziala, uzyj "LinguaBee - konsola.bat" - pokaze wszystkie komunikaty.
setlocal
set "ROOT=%~dp0"
set "MAIN=%ROOT%main.py"
if exist "%ROOT%app\main.py" set "MAIN=%ROOT%app\main.py"

set "PYW="
if exist "%ROOT%python\pythonw.exe" set "PYW=%ROOT%python\pythonw.exe"
if not defined PYW if exist "%ROOT%venv\Scripts\pythonw.exe" set "PYW=%ROOT%venv\Scripts\pythonw.exe"
if not defined PYW if exist "%ROOT%.venv\Scripts\pythonw.exe" set "PYW=%ROOT%.venv\Scripts\pythonw.exe"

if not defined PYW (
    echo Nie znaleziono srodowiska Pythona dla LinguaBee w:
    echo   %ROOT%
    echo.
    echo Najpierw zainstaluj biblioteki: kliknij dwukrotnie setup_windows.bat
    echo.
    pause
    exit /b 1
)

for %%I in ("%MAIN%") do cd /d "%%~dpI"
start "" "%PYW%" "%MAIN%" %*
