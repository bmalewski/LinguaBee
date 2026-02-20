param(
    [switch]$SkipVenv,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

Write-Host "== LinguaBee: setup Windows ==" -ForegroundColor Cyan
Write-Host "Katalog: $PSScriptRoot"

Push-Location $PSScriptRoot
try {
    # 1) Python check
    $pyCmd = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyCmd = "py -3.11"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pyCmd = "python"
    }

    if (-not $pyCmd) {
        Write-Host "Nie znaleziono Pythona w PATH." -ForegroundColor Red
        Write-Host "Zainstaluj Python 3.11.9: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -ForegroundColor Yellow
        exit 1
    }

    # 2) Git check
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "Nie znaleziono Git w PATH." -ForegroundColor Red
        Write-Host "Zainstaluj Git: https://git-scm.com/download/win" -ForegroundColor Yellow
        exit 1
    }

    # 3) venv
    if (-not $SkipVenv) {
        if (-not (Test-Path ".venv")) {
            Write-Host "Tworzenie środowiska .venv..." -ForegroundColor Green
            Invoke-Expression "$pyCmd -m venv .venv"
        } else {
            Write-Host "Środowisko .venv już istnieje." -ForegroundColor DarkYellow
        }
    }

    # 4) install deps
    if (-not $SkipInstall) {
        $venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            Write-Host "Brak .venv\Scripts\python.exe. Uruchom ponownie bez -SkipVenv." -ForegroundColor Red
            exit 1
        }

        Write-Host "Aktualizacja pip..." -ForegroundColor Green
        & $venvPython -m pip install --upgrade pip

        Write-Host "Instalacja zależności z requirements.txt..." -ForegroundColor Green
        & $venvPython -m pip install -r requirements.txt

        Write-Host "Aktualizacja yt-dlp (zalecane)..." -ForegroundColor Green
        & $venvPython -m pip install -U yt-dlp
    }

    Write-Host ""
    Write-Host "Setup zakończony." -ForegroundColor Cyan
    Write-Host "Uruchomienie aplikacji:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python main.py"
    Write-Host ""
    Write-Host "Jeśli brakuje ffmpeg, pobierz: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
    Write-Host "Jeśli masz problemy z YouTube, doinstaluj Node.js: https://nodejs.org/en/download" -ForegroundColor Yellow
}
finally {
    Pop-Location
}
