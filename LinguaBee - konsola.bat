@echo off
rem LinguaBee z oknem konsoli - do diagnozy: widac wszystkie komunikaty i bledy.
rem Okno zostaje otwarte po zamknieciu programu, zeby mozna bylo przeczytac komunikaty.
setlocal
set "ROOT=%~dp0"
set "MAIN=%ROOT%main.py"
if exist "%ROOT%app\main.py" set "MAIN=%ROOT%app\main.py"

set "PY="
if exist "%ROOT%python\python.exe" set "PY=%ROOT%python\python.exe"
if not defined PY if exist "%ROOT%venv\Scripts\python.exe" set "PY=%ROOT%venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"

if not defined PY (
    echo Nie znaleziono srodowiska Pythona dla LinguaBee w:
    echo   %ROOT%
    echo.
    echo Najpierw zainstaluj biblioteki: kliknij dwukrotnie setup_windows.bat
    echo.
    pause
    exit /b 1
)

for %%I in ("%MAIN%") do cd /d "%%~dpI"
echo Uruchamiam LinguaBee: "%PY%" "%MAIN%" %*
echo.
"%PY%" "%MAIN%" %*
echo.
echo LinguaBee zakonczyl dzialanie (kod wyjscia: %ERRORLEVEL%).
pause
