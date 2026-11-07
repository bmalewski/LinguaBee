<#
Buduje lekki instalator LinguaBee dla Windows: dist_installer\LinguaBee-<wersja>-Setup.exe (ok. 25 MB).

Instalator zawiera tylko kod programu i uv.exe. Python, PyTorch dobrany do karty graficznej,
biblioteki i ffmpeg pobiera przy instalacji packaging\installer\bootstrap.ps1.

Użycie (z katalogu projektu):
  powershell -ExecutionPolicy Bypass -File packaging\installer\build_installer.ps1
  powershell -ExecutionPolicy Bypass -File packaging\installer\build_installer.ps1 -StageOnly   # bez kompilacji

Wymaga: git, Inno Setup 6.7+ (winget install -e --id JRSoftware.InnoSetup), internet (uv.exe).
Wersja: APP_VERSION w config.py. Listy pakietów: packaging\installer\compile_locks.ps1.
#>
[CmdletBinding()]
param(
    [switch]$StageOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'downloads.json') | ConvertFrom-Json

$versionLine = Select-String -Path (Join-Path $Root 'config.py') -Pattern '^APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionLine) { throw 'Nie znaleziono APP_VERSION w config.py' }
$Version = $versionLine.Matches[0].Groups[1].Value
Write-Host "== LinguaBee ${Version}: budowanie instalatora ==" -ForegroundColor Cyan

foreach ($v in 'nvidia', 'cpu', 'amd') {
    if (-not (Test-Path (Join-Path $Root "requirements\windows-$v.txt"))) {
        throw "Brak requirements\windows-$v.txt - uruchom packaging\installer\compile_locks.ps1"
    }
}
$dirty = @(git -c core.quotepath=off status --porcelain)
if ($dirty.Count -gt 0) {
    Write-Host "UWAGA: w repozytorium są niezacommitowane zmiany ($($dirty.Count)) - trafią do instalatora w obecnej postaci." -ForegroundColor Yellow
}

# ------------------------------------------------------------------ katalog do budowania
$stage = Join-Path $Root 'build_assets\installer'
if (Test-Path $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
$stageApp = Join-Path $stage 'app'
$stageBin = Join-Path $stage 'bin'
$stageRoot = Join-Path $stage 'root'
foreach ($d in $stageApp, $stageBin, $stageRoot) { New-Item -ItemType Directory -Force $d | Out-Null }

# Pliki śledzone przez git oraz nowe, nieignorowane (bez user_settings.json z kluczami API,
# modeli, wyników, logów i venv - są w .gitignore). Bez plików tylko dla deweloperów.
$exclude = '^(\.git|Dockerfile$|\.dockerignore$|setup_windows\.|setup_macos\.sh$|Utworz skrot|LinguaBee( - konsola)?\.bat$|GITHUB_UPLOAD_CHECKLIST|WHAT_TO_COMMIT|packaging/(LinguaBee-macos\.spec|build_macos\.sh|build_pkg\.sh|pkg/))'
$files = @(git -c core.quotepath=off ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) { throw 'git ls-files nie powiódł się' }
$copied = 0
foreach ($f in $files) {
    if ($f -match $exclude) { continue }
    $src = Join-Path $Root $f
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { continue }  # usunięte, jeszcze nie zacommitowane
    $dst = Join-Path $stageApp $f
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst
    $copied++
}
Write-Host "  kod programu: $copied plików"

# Pliki do katalogu głównego instalacji (uruchamianie i naprawa).
Copy-Item -LiteralPath (Join-Path $Root 'LinguaBee.bat') -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $Root 'LinguaBee - konsola.bat') -Destination $stageRoot
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Napraw instalacje.bat') -Destination $stageRoot

# ------------------------------------------------------------------ uv.exe (przypięta wersja)
$dl = Join-Path $Root 'build_assets\downloads'
New-Item -ItemType Directory -Force $dl | Out-Null
$uvZip = Join-Path $dl (Split-Path $manifest.uv.url -Leaf)
if (-not (Test-Path $uvZip) -or (Get-FileHash $uvZip -Algorithm SHA256).Hash -ine $manifest.uv.sha256) {
    Write-Host "  pobieram uv $($manifest.uv.version)..."
    Invoke-WebRequest -UseBasicParsing -Uri $manifest.uv.url -OutFile $uvZip
}
if ((Get-FileHash $uvZip -Algorithm SHA256).Hash -ine $manifest.uv.sha256) { throw 'uv: niezgodna suma SHA-256' }
$archive = [IO.Compression.ZipFile]::OpenRead($uvZip)
try {
    $entry = $archive.Entries | Where-Object { $_.Name -ieq 'uv.exe' } | Select-Object -First 1
    if (-not $entry) { throw 'w archiwum uv brak uv.exe' }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $stageBin 'uv.exe'), $true)
}
finally { $archive.Dispose() }
Write-Host "  uv $($manifest.uv.version): OK"

if ($StageOnly) {
    Write-Host "Gotowe (bez kompilacji): $stage" -ForegroundColor Green
    return
}

# ------------------------------------------------------------------ Inno Setup
$iscc = $null
foreach ($cand in @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'))) {
    if ($cand -and (Test-Path $cand)) { $iscc = $cand; break }
}
if (-not $iscc) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}
if (-not $iscc) { throw 'Nie znaleziono ISCC.exe. Zainstaluj Inno Setup 6: winget install -e --id JRSoftware.InnoSetup' }

Write-Host "== Inno Setup: $iscc =="
& $iscc /Q "/DAppVersion=$Version" "/DStaging=$stage" (Join-Path $PSScriptRoot 'LinguaBee.iss')
if ($LASTEXITCODE -ne 0) { throw "Kompilacja instalatora nie powiodła się (ISCC, kod $LASTEXITCODE)" }

$setup = Join-Path $Root "dist_installer\LinguaBee-$Version-Setup.exe"
Write-Host ("Gotowe: {0} ({1:N1} MB)" -f $setup, ((Get-Item $setup).Length / 1MB)) -ForegroundColor Green
