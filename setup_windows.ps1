param(
    [switch]$SkipVenv,
    [switch]$SkipInstall,
    [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'

Write-Host "== LinguaBee: setup Windows ==" -ForegroundColor Cyan
Write-Host "Katalog: $PSScriptRoot"

Push-Location $PSScriptRoot
try {
    # 1) Python check
    $pyCmd = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyCmd = "py -3.12"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pyCmd = "python"
    }

    if (-not $pyCmd) {
        Write-Host "Nie znaleziono Pythona w PATH." -ForegroundColor Red
        Write-Host "Zainstaluj Python 3.12.10: https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -ForegroundColor Yellow
        exit 1
    }

    # 2) Git check
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "Nie znaleziono Git w PATH." -ForegroundColor Red
        Write-Host "Zainstaluj Git: https://git-scm.com/download/win" -ForegroundColor Yellow
        exit 1
    }

    # 3) venv: istniejące venv\ albo .venv\; nowe tworzymy jako .venv\
    $venvDir = @('venv', '.venv') | Where-Object { Test-Path (Join-Path $_ 'Scripts\python.exe') } | Select-Object -First 1
    if (-not $SkipVenv) {
        if (-not $venvDir) {
            $venvDir = '.venv'
            Write-Host "Tworzenie środowiska $venvDir..." -ForegroundColor Green
            Invoke-Expression "$pyCmd -m venv $venvDir"
        } else {
            Write-Host "Środowisko $venvDir już istnieje." -ForegroundColor DarkYellow
        }
    }
    if (-not $venvDir) { $venvDir = '.venv' }

    # 4) install deps
    if (-not $SkipInstall) {
        $venvPython = Join-Path (Get-Location) "$venvDir\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            Write-Host "Brak $venvDir\Scripts\python.exe. Uruchom ponownie bez -SkipVenv." -ForegroundColor Red
            exit 1
        }

        Write-Host "Aktualizacja pip..." -ForegroundColor Green
        & $venvPython -m pip install --upgrade pip

        Write-Host "Wykrywanie GPU i instalacja właściwego PyTorch (CUDA/ROCm/CPU)..." -ForegroundColor Green
        & $venvPython install_pytorch.py

        Write-Host "Instalacja zależności z requirements.txt..." -ForegroundColor Green
        & $venvPython -m pip install -r requirements.txt

        Write-Host "Aktualizacja yt-dlp (zalecane)..." -ForegroundColor Green
        & $venvPython -m pip install -U yt-dlp
    }

    # 5) skrót na pulpicie i w menu Start
    if (-not $NoShortcut) {
        Write-Host "Tworzenie skrótu LinguaBee na pulpicie i w menu Start..." -ForegroundColor Green
        try {
            & (Join-Path $PSScriptRoot 'packaging\windows\create_shortcut.ps1') -Root $PSScriptRoot
        } catch {
            Write-Host "Nie udało się utworzyć skrótu: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Setup zakończony." -ForegroundColor Cyan
    Write-Host "Uruchamianie LinguaBee:" -ForegroundColor Cyan
    Write-Host "  - dwuklik w plik LinguaBee.bat albo w skrót LinguaBee na pulpicie / w menu Start"
    Write-Host "  - LinguaBee - konsola.bat: to samo, ale z oknem komunikatów (gdy coś nie działa)"
    Write-Host "  - ręcznie: .\$venvDir\Scripts\Activate.ps1, potem python main.py"
    Write-Host ""
    Write-Host "Jeśli brakuje ffmpeg, pobierz: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
    Write-Host "Jeśli masz problemy z YouTube, doinstaluj Node.js: https://nodejs.org/en/download" -ForegroundColor Yellow
}
finally {
    Pop-Location
}
