<#
LinguaBee - instalacja, naprawa i aktualizacja środowiska programu (Windows).

Uruchamiają go: instalator (LinguaBee-<wersja>-Setup.exe) po skopiowaniu plików oraz
"Napraw instalacje.bat". Do folderu instalacji pobiera:
  python\  Python 3.12 (python-build-standalone, przez uv) + biblioteki z przypiętej listy
           app\requirements\windows-<wariant>.txt (dokładne wersje, sumy SHA-256)
  bin\     ffmpeg.exe, ffprobe.exe (+ deno.exe do YouTube)
i zapisuje install.json. Każdy krok sprawdza, co już jest, więc ponowne uruchomienie
tylko naprawia i uzupełnia instalację.

Parametry:
  -Root    folder instalacji (zawiera app\ i bin\uv.exe)
  -Gpu     auto | nvidia | amd | cpu | keep (keep = jak w install.json, a gdy go brak - auto)
  -Repair  tryb naprawy (po przeniesieniu folderu poprawia też skróty)
  -NoDeno  bez Deno (yt-dlp potrzebuje go do YouTube)

Kody wyjścia: 0 = gotowe, 2 = gotowe z ostrzeżeniami, 1 = błąd.
Postęp: linie "@@KROK n/7: ..." (czyta je instalator), pełny log: <root>\user\logs\install.log.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [ValidateSet('auto', 'nvidia', 'amd', 'cpu', 'keep')][string]$Gpu = 'keep',
    [switch]$Repair,
    [switch]$NoDeno
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Uruchomiony pośrednio z PowerShell 7 (instalator albo .bat otwarty w terminalu PS7) Windows PowerShell
# dziedziczy PSModulePath z modułami PS7 i nie widzi m.in. Get-FileHash, Get-CimInstance.
# Przywracamy domyślne ścieżki modułów Windows PowerShell (tylko dla tego procesu).
$env:PSModulePath = (@(
        [IO.Path]::Combine([Environment]::GetFolderPath('MyDocuments'), 'WindowsPowerShell\Modules'),
        [IO.Path]::Combine($env:ProgramFiles, 'WindowsPowerShell\Modules'),
        [IO.Path]::Combine($PSHOME, 'Modules')
    ) -join ';')
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root = [IO.Path]::GetFullPath($Root.Trim('"') + '\').TrimEnd('\')
$App = Join-Path $Root 'app'
$Bin = Join-Path $Root 'bin'
$PyDir = Join-Path $Root 'python'
$Py = Join-Path $PyDir 'python.exe'
$Cache = Join-Path $Root 'cache'
$Downloads = Join-Path $Cache 'downloads'
$LogDir = Join-Path $Root 'user\logs'
$LogFile = Join-Path $LogDir 'install.log'
$InstallJson = Join-Path $Root 'install.json'
$Uv = Join-Path $Bin 'uv.exe'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$TotalSteps = 7
$script:Warnings = 0

foreach ($d in $Bin, $Cache, $Downloads, $LogDir) { New-Item -ItemType Directory -Force $d | Out-Null }

# uv: cache na tym samym dysku co instalacja (pliki są podlinkowane, nie zdublowane),
# bez pasków postępu i kolorów (wyjście trafia do logu), Python wypisuje UTF-8.
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $Cache 'uv' }
$env:UV_NO_PROGRESS = '1'
$env:NO_COLOR = '1'
$env:PYTHONUTF8 = '1'

function Write-Log([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message
    [Console]::Out.WriteLine($line)
    try { [IO.File]::AppendAllText($LogFile, $line + "`r`n", $Utf8) } catch { }
}

function Write-Step([int]$Number, [string]$Message) {
    Write-Log ('@@KROK {0}/{1}: {2}' -f $Number, $TotalSteps, $Message)
}

function Write-Warn([string]$Message) {
    $script:Warnings++
    Write-Log "UWAGA: $Message"
}

function Invoke-Logged([string]$Exe, [string[]]$Arguments) {
    # Uruchamia program, przepisuje jego stdout/stderr do logu, zwraca kod wyjścia.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $text = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { [string]$_ }
            if ($text.Trim()) { Write-Log ('  ' + $text.TrimEnd()) }
        }
        return $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
}

function Get-VerifiedFile([string]$Url, [string]$Sha256, [string]$Destination) {
    # Pobiera plik (curl: ponawianie, wznawianie) i sprawdza sumę SHA-256.
    if (Test-Path $Destination) {
        if (-not $Sha256 -or (Get-FileHash $Destination -Algorithm SHA256).Hash -ieq $Sha256) {
            Write-Log "  już pobrany: $(Split-Path $Destination -Leaf)"
            return
        }
        Remove-Item -LiteralPath $Destination -Force
    }
    $part = "$Destination.part"
    Write-Log "  pobieram: $Url"
    $code = Invoke-Logged 'curl.exe' @('-sS', '-L', '--fail', '--retry', '5', '--retry-delay', '3', '-C', '-', '-o', $part, $Url)
    if ($code -ne 0) { throw "pobieranie nie powiodło się (curl, kod $code): $Url" }
    if ($Sha256) {
        $actual = (Get-FileHash $part -Algorithm SHA256).Hash
        if ($actual -ine $Sha256) {
            Remove-Item -LiteralPath $part -Force
            throw "niezgodna suma SHA-256 pliku $(Split-Path $Destination -Leaf)"
        }
    }
    Move-Item -LiteralPath $part -Destination $Destination -Force
}

function Expand-ZipEntries([string]$Zip, [string[]]$Names, [string]$TargetDir) {
    # Wypakowuje z archiwum ZIP tylko wskazane pliki (po nazwie, z dowolnego podkatalogu).
    $archive = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        foreach ($name in $Names) {
            $entry = $archive.Entries | Where-Object { $_.Name -ieq $name } | Select-Object -First 1
            if (-not $entry) { throw "w archiwum $(Split-Path $Zip -Leaf) brak pliku $name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $TargetDir $name), $true)
        }
    }
    finally { $archive.Dispose() }
}

function Test-ReparsePoint($Item) {
    return [bool]($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Remove-DirectorySafe([string]$Path) {
    # "rmdir /s" z cmd usuwa dowiązania (junction) uv bez wchodzenia do katalogów, na które
    # wskazują; Remove-Item -Recurse w PowerShell 5.1 potrafi za nimi podążać.
    # Get-Item -Force widzi też zerwane dowiązanie (Test-Path już nie).
    if (-not (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue)) { return }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'  # w PS 5.1 stderr programu przy 'Stop' kończy skrypt
    try { & cmd.exe /d /c "rmdir /s /q `"$Path`"" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $previous }
    if (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue) { throw "nie udało się usunąć $Path" }
}

function Get-PythonVersion {
    if (-not (Test-Path $Py)) { return $null }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]($out | Select-Object -First 1)).Trim() }
    }
    catch { }
    finally { $ErrorActionPreference = $previous }
    return $null
}

function Get-ToolVersionLine([string]$Exe, [string[]]$Arguments) {
    if (-not (Test-Path $Exe)) { return '' }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { return [string](& $Exe @Arguments 2>$null | Select-Object -First 1) }
    catch { return '' }
    finally { $ErrorActionPreference = $previous }
}

function Get-BrokenPackages {
    # Nazwy pakietów z brakującymi lub uciętymi plikami (verify_packages.py); pusta lista, gdy nie da się sprawdzić.
    $verify = Join-Path $PSScriptRoot 'verify_packages.py'
    if (-not (Test-Path $Py) -or -not (Test-Path $verify)) { return @() }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Py -I $verify 2>$null
        if ($LASTEXITCODE -ne 0) { return @() }
        return @($out | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -match '^[A-Za-z0-9][A-Za-z0-9._-]*$' })
    }
    catch { return @() }
    finally { $ErrorActionPreference = $previous }
}

function Read-InstallInfo {
    if (Test-Path $InstallJson) {
        try { return (Get-Content -Raw -Encoding UTF8 $InstallJson | ConvertFrom-Json) } catch { }
    }
    return $null
}

function Get-GpuNames {
    try { return @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { [string]$_.Name }) }
    catch { return @() }
}

function Get-AutoVariant([string[]]$Names) {
    # NVIDIA -> CUDA. AMD tylko dla kart wspieranych przez PyTorch ROCm na Windows
    # (RDNA3/RDNA4, Ryzen AI Max / AI 300), inne (np. starsze iGPU) -> CPU.
    $joined = ($Names -join ' | ')
    if ($joined -match 'NVIDIA') { return 'nvidia' }
    if ($joined -match 'Radeon(\(TM\))? RX (7|9)\d{3}|Radeon(\(TM\))? (AI )?PRO (W7|R9)|Radeon(\(TM\))? 80[56]0S|Radeon(\(TM\))? 8[89]0M') { return 'amd' }
    return 'cpu'
}

function Update-MovedShortcuts([string]$OldRoot) {
    # Po przeniesieniu folderu: skróty z instalatora (menu Start\LinguaBee, pulpit) wskazujące
    # stary folder dostają nową ścieżkę. Zwraca liczbę skrótów, które wskazują teraz ten folder.
    $pattern = '(?i)' + [regex]::Escape($OldRoot.TrimEnd('\')) + '(?=\\|"|$)'
    $current = '(?i)' + [regex]::Escape($Root) + '(?=\\|"|$)'
    $replacement = $Root.Replace('$', '$$')
    $shell = New-Object -ComObject WScript.Shell
    $count = 0
    $places = @((Join-Path ([Environment]::GetFolderPath('Programs')) 'LinguaBee'), [Environment]::GetFolderPath('Desktop'))
    foreach ($place in $places) {
        if (-not $place -or -not (Test-Path -LiteralPath $place)) { continue }
        foreach ($file in @(Get-ChildItem -LiteralPath $place -Filter '*.lnk' -File -ErrorAction SilentlyContinue)) {
            $link = $shell.CreateShortcut($file.FullName)
            if ($link.TargetPath -match $current) { $count++; continue }  # np. świeży skrót z instalatora
            if ($link.TargetPath -notmatch $pattern) { continue }
            $link.TargetPath = $link.TargetPath -replace $pattern, $replacement
            $link.Arguments = $link.Arguments -replace $pattern, $replacement
            $link.WorkingDirectory = $link.WorkingDirectory -replace $pattern, $replacement
            $link.IconLocation = $link.IconLocation -replace $pattern, $replacement
            $link.Save()
            Write-Log "  poprawiony skrót: $($file.FullName)"
            $count++
        }
    }
    return $count
}

$info = Read-InstallInfo
$mode = if ($Repair) { 'naprawa' } else { 'instalacja' }
Write-Log "=== LinguaBee: $mode w $Root ==="

try {
    if (-not (Test-Path $Uv)) { throw "brak $Uv - uruchom ponownie instalator LinguaBee." }
    if (-not (Test-Path (Join-Path $App 'main.py'))) { throw "brak kodu programu w $App - uruchom ponownie instalator LinguaBee." }
    $manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'downloads.json') | ConvertFrom-Json

    # ------------------------------------------------------------------ 1. system
    Write-Step 1 'Sprawdzanie systemu'
    if (-not [Environment]::Is64BitOperatingSystem -or [Environment]::OSVersion.Version.Major -lt 10) {
        throw 'LinguaBee wymaga 64-bitowego Windows 10 lub 11.'
    }
    $lowerRoot = $Root.ToLowerInvariant()
    foreach ($forbidden in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
        if ($forbidden -and $lowerRoot.StartsWith($forbidden.ToLowerInvariant())) {
            throw "LinguaBee nie może być zainstalowany w $forbidden (Program Files albo OneDrive). Wybierz np. C:\LinguaBee albo D:\LinguaBee."
        }
    }
    $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($Root))
    if ($drive.DriveFormat -match '^FAT') {
        throw "dysk $($drive.Name) ma system plików $($drive.DriveFormat) (limit 4 GB na plik) - wybierz dysk NTFS."
    }
    # Uruchomiony program blokuje pliki bibliotek (np. DLL PyTorch) - naprawa mogłaby się nie udać.
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($PyDir + '\', [StringComparison]::OrdinalIgnoreCase) })
    if ($running.Count -gt 0) { throw 'LinguaBee jest uruchomiony z tego folderu - zamknij program i spróbuj ponownie.' }
    $gpuNames = Get-GpuNames
    Write-Log ('Karty graficzne: ' + $(if ($gpuNames) { $gpuNames -join '; ' } else { 'nie wykryto' }))

    # ------------------------------------------------------------------ 2. wariant
    Write-Step 2 'Wybór wariantu (karta graficzna)'
    $variant = $Gpu
    if ($variant -eq 'keep') {
        $variant = if ($info -and $info.variant -in @('nvidia', 'amd', 'cpu')) { [string]$info.variant } else { 'auto' }
    }
    if ($variant -eq 'auto') { $variant = Get-AutoVariant $gpuNames }
    Write-Log "Wariant: $variant"
    $lock = Join-Path $App "requirements\windows-$variant.txt"
    if (-not (Test-Path $lock)) { throw "brak listy pakietów $lock" }

    # Miejsce sprawdzamy przy pierwszej instalacji i zmianie karty; aktualizacja i naprawa pobierają niewiele.
    $fresh = (-not (Test-Path $Py)) -or (-not $info) -or ([string]$info.variant -ne $variant)
    if ($fresh) {
        $needGb = [double]$manifest.space_gb.$variant
        # Bez NTFS (np. exFAT) uv nie może łączyć plików z cache, tylko je kopiuje: w trakcie instalacji 2x więcej.
        if ($drive.DriveFormat -ne 'NTFS') { $needGb *= 2 }
        $freeGb = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
        Write-Log "Wolne miejsce na $($drive.Name) ($($drive.DriveFormat)): $freeGb GB (potrzeba ok. $needGb GB, bez modeli)"
        if ($freeGb -lt $needGb) { throw "za mało miejsca na dysku $($drive.Name): wolne $freeGb GB, potrzeba ok. $needGb GB." }
    }

    $sys32 = Join-Path $env:SystemRoot 'System32'
    $missingVc = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') | Where-Object { -not (Test-Path (Join-Path $sys32 $_)) }
    if ($missingVc) {
        Write-Log "Brak bibliotek Visual C++ ($($missingVc -join ', ')) - instaluję VC++ Redistributable (Windows poprosi o zgodę)."
        try {
            $vc = Join-Path $Downloads 'vc_redist.x64.exe'
            Get-VerifiedFile $manifest.vc_redist.url $null $vc
            $sig = Get-AuthenticodeSignature $vc
            if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Microsoft') { throw 'nieprawidłowy podpis pliku vc_redist.x64.exe' }
            $p = Start-Process -FilePath $vc -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -Wait -PassThru
            if ($p.ExitCode -notin @(0, 1638, 3010)) { Write-Warn "instalator VC++ zakończył się kodem $($p.ExitCode)" }
        }
        catch { Write-Warn "nie udało się zainstalować VC++ Redistributable: $($_.Exception.Message)" }
    }

    if ($variant -eq 'nvidia') {
        $driver = Get-ToolVersionLine (Join-Path $sys32 'nvidia-smi.exe') @('--query-gpu=driver_version', '--format=csv,noheader')
        if ($driver -match '^\s*(\d+)') {
            if ([int]$Matches[1] -lt 570) { Write-Warn "sterownik NVIDIA $($driver.Trim()) jest za stary dla PyTorch CUDA 12.8 - zainstaluj wersję 570 lub nowszą." }
            else { Write-Log "Sterownik NVIDIA: $($driver.Trim())" }
        }
    }

    # ------------------------------------------------------------------ 3. Python
    Write-Step 3 "Python $($manifest.python.version)"
    $pyVersion = Get-PythonVersion
    if ($pyVersion -eq $manifest.python.version) {
        Write-Log "Python $pyVersion jest już zainstalowany."
    }
    else {
        if ($pyVersion) { Write-Log "Zmiana wersji Pythona: $pyVersion -> $($manifest.python.version) (biblioteki zostaną zainstalowane ponownie)." }
        $tmp = Join-Path $Cache 'uv-python'
        Remove-DirectorySafe $tmp
        $code = Invoke-Logged $Uv @('python', 'install', $manifest.python.version, '--install-dir', $tmp, '--no-bin', '--no-registry', '--no-config')
        if ($code -ne 0) { throw "pobieranie Pythona nie powiodło się (uv, kod $code)" }
        # uv tworzy obok katalogu Pythona dowiązanie "cpython-3.12-..." - bierzemy prawdziwy katalog.
        $src = Get-ChildItem -LiteralPath $tmp -Directory -Force |
            Where-Object { -not (Test-ReparsePoint $_) -and (Test-Path (Join-Path $_.FullName 'python.exe')) } |
            Select-Object -First 1
        if (-not $src) { throw 'nie znaleziono pobranego Pythona' }
        Remove-DirectorySafe $PyDir
        Move-Item -LiteralPath $src.FullName -Destination $PyDir
        Remove-DirectorySafe $tmp
    }
    # uv oznacza swoje Pythony jako "zarządzane zewnętrznie"; ten jest prywatny dla LinguaBee.
    $marker = Join-Path $PyDir 'Lib\EXTERNALLY-MANAGED'
    if (Test-Path $marker) { Remove-Item -LiteralPath $marker -Force }
    $pyVersion = Get-PythonVersion
    if (-not $pyVersion) { throw "Python w $PyDir nie uruchamia się" }

    # ------------------------------------------------------------------ 4. biblioteki
    Write-Step 4 "Biblioteki i PyTorch ($variant) - największe pobieranie, może potrwać kilkanaście minut"
    # uv uznaje pakiet za zainstalowany po samych metadanych; usunięte lub ucięte pliki wykrywa
    # verify_packages.py (lista RECORD) - takie pakiety instalujemy ponownie.
    $broken = @(Get-BrokenPackages)
    if ($broken.Count -gt 0) { Write-Log "Uszkodzone pakiety (zostaną zainstalowane ponownie): $($broken -join ', ')" }
    $reinstall = @($broken | Where-Object { $_ -ne 'yt-dlp' } | ForEach-Object { '--reinstall-package', $_ })
    $code = Invoke-Logged $Uv (@('pip', 'install', '--no-config', '--python', $Py, '--index-strategy', 'unsafe-best-match', '-r', $lock) + $reinstall)
    if ($code -ne 0) { throw "instalacja bibliotek nie powiodła się (uv, kod $code)" }

    if ($variant -eq 'amd') {
        # CTranslate2 z ROCm (Whisper i NLLB na karcie AMD); paczka z PyPI liczy tylko na CPU.
        try {
            $ct = $manifest.ctranslate2_rocm
            $zip = Join-Path $Downloads "ctranslate2-$($ct.version)-rocm-windows.zip"
            Get-VerifiedFile $ct.url $ct.sha256 $zip
            Expand-ZipEntries $zip @($ct.wheel) $Downloads
            $code = Invoke-Logged $Uv @('pip', 'install', '--no-config', '--python', $Py, '--no-deps', '--reinstall', (Join-Path $Downloads $ct.wheel))
            if ($code -ne 0) { throw "uv, kod $code" }
            Write-Log "CTranslate2 $($ct.version) z ROCm zainstalowany."
        }
        catch { Write-Warn "CTranslate2 z ROCm nie zainstalował się ($($_.Exception.Message)) - Whisper i NLLB będą liczyć na CPU." }
    }

    # yt-dlp nie jest w liście pakietów: YouTube często się zmienia, więc zawsze najnowsza wersja.
    $ytArgs = @('pip', 'install', '--no-config', '--python', $Py, '--upgrade', 'yt-dlp')
    if ($broken -contains 'yt-dlp') { $ytArgs += @('--reinstall-package', 'yt-dlp') }
    $code = Invoke-Logged $Uv $ytArgs
    if ($code -ne 0) {
        if (-not (Test-Path (Join-Path $PyDir 'Lib\site-packages\yt_dlp\__init__.py'))) {
            throw "instalacja yt-dlp nie powiodła się (uv, kod $code)"
        }
        Write-Warn 'nie udało się zaktualizować yt-dlp (zostaje dotychczasowa wersja).'
    }

    # Cache uv jest połączony z python\ dowiązaniami twardymi: nie zajmuje miejsca drugi raz, ale
    # Eksplorator pokazuje podwójny rozmiar, a kopia folderu na inny dysk naprawdę go podwaja.
    # Biblioteki są już zainstalowane, więc cache nie jest potrzebny.
    Write-Log 'Czyszczenie cache pobranych paczek...'
    $code = Invoke-Logged $Uv @('cache', 'clean', '--no-config')
    if ($code -ne 0) { Write-Log 'Nie udało się wyczyścić cache uv (nie wpływa na działanie programu).' }

    # ------------------------------------------------------------------ 5. ffmpeg
    $ff = $manifest.ffmpeg
    Write-Step 5 "ffmpeg $($ff.version)"
    $ffmpeg = Join-Path $Bin 'ffmpeg.exe'
    if ((Get-ToolVersionLine $ffmpeg @('-hide_banner', '-version')) -match [regex]::Escape($ff.version) -and (Test-Path (Join-Path $Bin 'ffprobe.exe'))) {
        Write-Log "ffmpeg $($ff.version) jest już zainstalowany."
    }
    else {
        try {
            $zip = Join-Path $Downloads (Split-Path $ff.url -Leaf)
            Get-VerifiedFile $ff.url $ff.sha256 $zip
            Expand-ZipEntries $zip @('ffmpeg.exe', 'ffprobe.exe') $Bin
            Remove-Item -LiteralPath $zip -Force
            Write-Log "ffmpeg $($ff.version) zainstalowany."
        }
        catch { Write-Warn "ffmpeg: $($_.Exception.Message). Bez ffmpeg nie działa konwersja do MP3 i część przetwarzania audio." }
    }

    # ------------------------------------------------------------------ 6. Deno (YouTube)
    $deno = $manifest.deno
    Write-Step 6 "Deno $($deno.version) (YouTube)"
    $denoExe = Join-Path $Bin 'deno.exe'
    if ($NoDeno) {
        Write-Log 'Pominięto (bez obsługi YouTube przez Deno).'
    }
    elseif ((Get-ToolVersionLine $denoExe @('--version')) -match [regex]::Escape($deno.version)) {
        Write-Log "Deno $($deno.version) jest już zainstalowany."
    }
    else {
        try {
            $zip = Join-Path $Downloads (Split-Path $deno.url -Leaf)
            Get-VerifiedFile $deno.url $deno.sha256 $zip
            Expand-ZipEntries $zip @('deno.exe') $Bin
            Remove-Item -LiteralPath $zip -Force
            Write-Log "Deno $($deno.version) zainstalowany."
        }
        catch { Write-Warn "Deno: $($_.Exception.Message). Pobieranie z YouTube może nie działać." }
    }

    # ------------------------------------------------------------------ 7. zakończenie
    Write-Step 7 'Zapis konfiguracji i test programu'
    $appVersion = '?'
    $versionLine = Select-String -Path (Join-Path $App 'config.py') -Pattern '^APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($versionLine) { $appVersion = $versionLine.Matches[0].Groups[1].Value }

    $data = [ordered]@{}
    if ($info) { foreach ($p in $info.PSObject.Properties) { $data[$p.Name] = $p.Value } }  # zachowuje np. models_dir
    $movedFrom = if ($info -and $info.root -and ([string]$info.root -ne $Root)) { [string]$info.root } else { $null }
    $data['version'] = $appVersion
    $data['variant'] = $variant
    $data['python'] = $pyVersion
    $data['lock'] = "requirements/windows-$variant.txt"
    $data['lock_sha256'] = (Get-FileHash $lock -Algorithm SHA256).Hash.ToLowerInvariant()
    $data['uv'] = $manifest.uv.version
    $data['root'] = $Root
    if (-not $data.Contains('installed_at')) { $data['installed_at'] = (Get-Date -Format s) }
    $data['updated_at'] = (Get-Date -Format s)
    [IO.File]::WriteAllText($InstallJson, ($data | ConvertTo-Json -Depth 4), $Utf8)
    Write-Log "Zapisano $InstallJson"

    if ($movedFrom) {
        Write-Log "Folder instalacji został przeniesiony ($movedFrom -> $Root) - poprawiam skróty."
        $fixed = 0
        try { $fixed = Update-MovedShortcuts $movedFrom }
        catch { Write-Warn "nie udało się poprawić skrótów: $($_.Exception.Message)" }
        if ($fixed -eq 0) {
            # Brak skrótów (np. folder skopiowany z innego komputera) - nowy skrót w menu Start\LinguaBee.
            try { & (Join-Path $App 'packaging\windows\create_shortcut.ps1') -Root $Root -NoDesktop -StartMenuFolder 'LinguaBee' *>&1 | ForEach-Object { Write-Log "  $_" } }
            catch { Write-Warn "nie udało się utworzyć skrótu: $($_.Exception.Message)" }
        }
    }

    $code = Invoke-Logged $Py @((Join-Path $App 'main.py'), '--selftest')
    if ($code -ne 0) { Write-Warn 'test programu (--selftest) zgłosił problemy - szczegóły wyżej.' }
}
catch {
    Write-Log "BŁĄD: $($_.Exception.Message)"
    Write-Log 'Instalacja nie została dokończona. Usuń przyczynę opisaną wyżej (najczęściej brak internetu) i uruchom "Napraw instalację" (menu Start albo plik "Napraw instalacje.bat").'
    Write-Log "Pełny log: $LogFile"
    exit 1
}

if ($script:Warnings -gt 0) {
    Write-Log "Gotowe, ostrzeżeń: $($script:Warnings). Pełny log: $LogFile"
    exit 2
}
Write-Log "Gotowe. LinguaBee jest zainstalowany w $Root"
exit 0
