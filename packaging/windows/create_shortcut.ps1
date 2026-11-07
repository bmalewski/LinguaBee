<#
Tworzy skrót "LinguaBee" na pulpicie i w menu Start (Windows).

Skrót uruchamia pythonw.exe bezpośrednio (bez okna konsoli), z ikoną programu.
Python jest szukany w: python\ (instalator), venv\ albo .venv\ (instalacja ze źródeł).

Użycie:
  powershell -ExecutionPolicy Bypass -File packaging\windows\create_shortcut.ps1
  powershell -ExecutionPolicy Bypass -File packaging\windows\create_shortcut.ps1 -Root D:\LinguaBee -NoDesktop
  -StartMenuFolder LinguaBee: skrót w podfolderze menu Start (jak z instalatora) zamiast w głównym
#>
param(
    [string]$Root = (Join-Path $PSScriptRoot '..\..'),
    [switch]$NoDesktop,
    [switch]$NoStartMenu,
    [string]$StartMenuFolder
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path.TrimEnd('\')

$main = Join-Path $Root 'main.py'
if (-not (Test-Path $main)) {
    $main = Join-Path $Root 'app\main.py'
}
if (-not (Test-Path $main)) {
    Write-Host "Nie znaleziono main.py w $Root" -ForegroundColor Red
    exit 1
}

$pythonw = @('python\pythonw.exe', 'venv\Scripts\pythonw.exe', '.venv\Scripts\pythonw.exe') |
    ForEach-Object { Join-Path $Root $_ } |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1
if (-not $pythonw) {
    Write-Host "Nie znaleziono środowiska Pythona (venv) w $Root." -ForegroundColor Red
    Write-Host "Najpierw zainstaluj biblioteki: kliknij dwukrotnie setup_windows.bat" -ForegroundColor Yellow
    exit 1
}

$appDir = Split-Path $main
$icon = Join-Path $appDir 'icons\LinguaBee.ico'
if (-not (Test-Path $icon)) { $icon = Join-Path $appDir 'icons\icon.ico' }

$folders = @()
if (-not $NoDesktop) { $folders += [Environment]::GetFolderPath('Desktop') }
if (-not $NoStartMenu) {
    $programs = [Environment]::GetFolderPath('Programs')
    if ($StartMenuFolder) {
        $programs = Join-Path $programs $StartMenuFolder
        New-Item -ItemType Directory -Force $programs | Out-Null
    }
    $folders += $programs
}

$shell = New-Object -ComObject WScript.Shell
foreach ($folder in $folders) {
    $path = Join-Path $folder 'LinguaBee.lnk'
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $pythonw
    $link.Arguments = '"' + $main + '"'
    $link.WorkingDirectory = $appDir
    if (Test-Path $icon) { $link.IconLocation = "$icon,0" }
    $link.Description = 'LinguaBee - transkrypcja, tłumaczenie i streszczenia'
    $link.Save()
    Write-Host "Utworzono skrót: $path" -ForegroundColor Green
}
Write-Host "Python: $pythonw"
Write-Host "Program: $main"
