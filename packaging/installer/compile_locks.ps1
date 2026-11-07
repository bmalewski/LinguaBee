<#
Generuje przypięte listy pakietów (lockfile) dla instalatora Windows:
  requirements\windows-nvidia.txt  - PyTorch CUDA 12.8 (karty NVIDIA), z sumami SHA-256
  requirements\windows-cpu.txt     - PyTorch CPU, z sumami SHA-256
  requirements\windows-amd.txt     - PyTorch ROCm 7.2.1 od AMD (eksperymentalnie, bez sum)

Wersje pakietów są brane z działającego środowiska (venv), żeby instalator instalował dokładnie
przetestowany zestaw. Po zmianie requirements.txt albo aktualizacji venv uruchom skrypt ponownie.
Wyjątek: yt-dlp nie jest przypinany - bootstrap.ps1 instaluje zawsze jego najnowszą wersję.
Wzorowane na "compiled requirements" z ComfyUI Desktop.

Użycie (z katalogu projektu):
  powershell -ExecutionPolicy Bypass -File packaging\installer\compile_locks.ps1
  powershell -ExecutionPolicy Bypass -File packaging\installer\compile_locks.ps1 -Variant nvidia
Wymaga: uv w PATH (winget install astral-sh.uv) i działającego venv z zainstalowanymi pakietami.
#>
param(
    [ValidateSet('all', 'nvidia', 'cpu', 'amd')]
    [string]$Variant = 'all',
    [string]$Venv = 'venv'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw 'Nie znaleziono uv w PATH (winget install astral-sh.uv).' }
$py = Join-Path $Root "$Venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Brak środowiska: $py" }

$utf8 = New-Object System.Text.UTF8Encoding($false)
$work = Join-Path $Root 'build_assets\locks'
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root 'requirements') | Out-Null

# Wersje z działającego venv jako ograniczenia (-c): lockfile = przetestowany zestaw.
$freezeLines = @(& $py -m pip freeze --exclude-editable)
if ($LASTEXITCODE -ne 0 -or $freezeLines.Count -eq 0) { throw 'pip freeze nie powiódł się.' }
$freeze = Join-Path $work 'venv-freeze.txt'
[IO.File]::WriteAllLines($freeze, [string[]]$freezeLines, $utf8)
# Bez torch* - każdy wariant ma własny build PyTorch, przypięty przez --override poniżej.
$freezeNoTorch = Join-Path $work 'venv-freeze-no-torch.txt'
[IO.File]::WriteAllLines($freezeNoTorch, [string[]]($freezeLines | Where-Object { $_ -notmatch '^(torch|torchaudio|torchvision)==' }), $utf8)

$amdBase = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1'
$variants = [ordered]@{
    nvidia = @{
        Index     = 'https://download.pytorch.org/whl/cu128'
        Overrides = @('torch==2.8.0+cu128', 'torchaudio==2.8.0+cu128', 'torchvision==0.23.0+cu128')
        Hashes    = $true
    }
    cpu    = @{
        Index     = 'https://download.pytorch.org/whl/cpu'
        Overrides = @('torch==2.8.0+cpu', 'torchaudio==2.8.0+cpu', 'torchvision==0.23.0+cpu')
        Hashes    = $true
    }
    amd    = @{
        # Stabilne paczki AMD dla Windows (jak w ComfyUI Desktop); wymagają sterownika Adrenalin >= 26.2.2.
        Index     = $null
        Overrides = @(
            "torch @ $amdBase/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl",
            "torchaudio @ $amdBase/torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl",
            "torchvision @ $amdBase/torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"
        )
        # Biblioteki uruchomieniowe ROCm SDK (bez rocm-sdk-devel), jak w ComfyUI Desktop.
        Extra     = @(
            "rocm-sdk-core @ $amdBase/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
            "rocm-sdk-libraries-custom @ $amdBase/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
            "rocm @ $amdBase/rocm-7.2.1.tar.gz"
        )
        Hashes    = $false
    }
}

foreach ($name in $variants.Keys) {
    if ($Variant -ne 'all' -and $Variant -ne $name) { continue }
    $v = $variants[$name]
    $overrides = Join-Path $work "overrides-$name.txt"
    [IO.File]::WriteAllLines($overrides, [string[]]$v.Overrides, $utf8)
    $out = Join-Path $Root "requirements\windows-$name.txt"

    $inputs = @('requirements.txt')
    if ($v.Extra) {
        $extra = Join-Path $work "extra-$name.txt"
        [IO.File]::WriteAllLines($extra, [string[]]$v.Extra, $utf8)
        $inputs += $extra
    }

    $uvArgs = @('pip', 'compile') + $inputs + @(
        '--quiet',
        '-c', $freezeNoTorch,
        '--override', $overrides,
        '--index-strategy', 'unsafe-best-match',
        '--index-url', 'https://pypi.org/simple',
        '--python-version', '3.12',
        '--python-platform', 'x86_64-pc-windows-msvc',
        # --emit-index-url: lista sama wskazuje indeks PyTorch (torch==...+cu128 nie ma na PyPI)
        '--emit-index-url', '--emit-index-annotation', '--no-annotate',
        # yt-dlp zawsze w najnowszej wersji (YouTube często się zmienia) - instaluje go bootstrap.ps1.
        '--no-emit-package', 'yt-dlp',
        '--custom-compile-command', 'packaging\installer\compile_locks.ps1',
        '-o', $out
    )
    if ($v.Index) { $uvArgs += @('--extra-index-url', $v.Index) }
    if ($v.Hashes) { $uvArgs += '--generate-hashes' }

    Write-Host "== $name ==" -ForegroundColor Cyan
    & $uv @uvArgs
    if ($LASTEXITCODE -ne 0) { throw "uv pip compile ($name) zakończył się kodem $LASTEXITCODE" }
    $lines = Get-Content $out
    $pkgs = @($lines | Where-Object { $_ -match '^[A-Za-z0-9][A-Za-z0-9._-]*(==| @ )' }).Count
    $hashes = @($lines | Where-Object { $_ -match '--hash=sha256:' }).Count
    Write-Host ("  {0}: {1} pakietów, {2} sum SHA-256" -f (Split-Path $out -Leaf), $pkgs, $hashes) -ForegroundColor Green
}
