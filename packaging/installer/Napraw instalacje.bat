@echo off
rem LinguaBee - naprawa i uzupelnienie instalacji: Python, biblioteki, ffmpeg, Deno.
rem Sprawdza kazdy skladnik i pobiera tylko to, czego brakuje albo co jest uszkodzone.
rem Zmiana karty graficznej, np.:  "Napraw instalacje.bat" -Gpu cpu   (auto, nvidia, amd, cpu)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\packaging\installer\bootstrap.ps1" -Root "%~dp0." -Repair %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo Gotowe. Uruchom LinguaBee skrotem albo plikiem LinguaBee.bat
) else if "%RC%"=="2" (
    echo Gotowe, z ostrzezeniami - szczegoly powyzej i w pliku user\logs\install.log
) else (
    echo Naprawa nie powiodla sie - szczegoly powyzej i w pliku user\logs\install.log
)
pause
exit /b %RC%
