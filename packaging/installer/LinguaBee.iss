; LinguaBee - lekki instalator dla Windows (Inno Setup 6.7+), w stylu ComfyUI Desktop.
;
; Instalator zawiera tylko kod programu i uv.exe (ok. 25 MB). Python, PyTorch dobrany do karty
; graficznej, biblioteki (przypięte wersje, sumy SHA-256) i ffmpeg pobiera z internetu skrypt
; packaging\installer\bootstrap.ps1 - wszystko do jednego folderu wybranego przez użytkownika.
;
; Budowanie: powershell -ExecutionPolicy Bypass -File packaging\installer\build_installer.ps1
; Instalacja cicha (testy): LinguaBee-<wersja>-Setup.exe /VERYSILENT /DIR=D:\LinguaBee /GPU=nvidia
;   /GPU=auto|nvidia|amd|cpu, zadania: /TASKS="youtube" (bez ikony na pulpicie) itp.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Staging
  #define Staging "..\..\build_assets\installer"
#endif

[Setup]
AppId={{9C9163A7-5350-4F4A-91D1-0B492EE28371}
AppName=LinguaBee
AppVersion={#AppVersion}
AppVerName=LinguaBee {#AppVersion}
AppPublisher=Benedykt Malewski
AppPublisherURL=https://github.com/bmalewski/LinguaBee
AppSupportURL=https://github.com/bmalewski/LinguaBee
DefaultDirName={sd}\LinguaBee
DefaultGroupName=LinguaBee
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
UsePreviousAppDir=yes
DirExistsWarning=no
AllowRootDirectory=no
OutputDir=..\..\dist_installer
OutputBaseFilename=LinguaBee-{#AppVersion}-Setup
SetupIconFile=..\..\icons\LinguaBee.ico
UninstallDisplayIcon={app}\app\icons\LinguaBee.ico
UninstallDisplayName=LinguaBee
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ShowLanguageDialog=no
SetupLogging=yes
; Inno 6.5+ domyślnie włącza RedirectionGuard, który dziedziczą procesy potomne. Blokuje on
; dowiązania katalogów tworzone przez uv przy instalacji Pythona (błąd 448 "niezaufany punkt
; instalacji"). Instalator działa bez uprawnień administratora, więc ochrona nie jest potrzebna.
RedirectionGuard=no

[Languages]
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"

[Tasks]
Name: "desktopicon"; Description: "Ikona LinguaBee na pulpicie"; GroupDescription: "Dodatkowo:"
Name: "youtube"; Description: "Obsługa pobierania z YouTube (Deno, ok. 45 MB)"; GroupDescription: "Dodatkowo:"

[InstallDelete]
; Kod programu jest podmieniany w całości (usunięte pliki nie zostają po aktualizacji).
; Dane użytkownika (models, output, user) leżą poza app\ i zostają.
Type: filesandordirs; Name: "{app}\app"

[Files]
Source: "{#Staging}\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Staging}\bin\uv.exe"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Staging}\root\*"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\LinguaBee"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\icons\LinguaBee.ico"; Comment: "Transkrypcja, tłumaczenie i streszczenia"
Name: "{group}\LinguaBee (konsola)"; Filename: "{app}\LinguaBee - konsola.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app\icons\LinguaBee.ico"; Comment: "LinguaBee z oknem komunikatów (diagnoza)"
Name: "{group}\Napraw instalację"; Filename: "{app}\Napraw instalacje.bat"; WorkingDir: "{app}"; Comment: "Sprawdza i uzupełnia Pythona, biblioteki i ffmpeg"
Name: "{group}\Folder LinguaBee"; Filename: "{app}"; Comment: "Modele, wyniki i ustawienia"
Name: "{group}\Odinstaluj LinguaBee"; Filename: "{uninstallexe}"
Name: "{autodesktop}\LinguaBee"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\icons\LinguaBee.ico"; Comment: "Transkrypcja, tłumaczenie i streszczenia"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; Description: "Uruchom LinguaBee"; Flags: nowait postinstall skipifsilent; Check: BootstrapSucceeded

[UninstallDelete]
; Pobrane przez bootstrap.ps1 (instalator ich nie kopiował, więc usuwamy je jawnie).
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{app}\install.json"

[Code]
const
  BootstrapSteps = 7;
  VariantNvidia = 0;
  VariantAmd = 1;
  VariantCpu = 2;

var
  GpuPage: TInputOptionWizardPage;
  DetectedGpus: String;
  BootstrapResult: Integer;
  DeleteUserData: Boolean;

{ ---------------------------------------------------------------- karta graficzna }

function GetVideoControllers(): String;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
begin
  Result := '';
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT Name FROM Win32_VideoController');
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      if Result <> '' then
        Result := Result + '; ';
      Result := Result + Item.Name;
    end;
  except
    Result := '';
  end;
end;

function IsSupportedAmd(const Names: String): Boolean;
var
  L: String;
begin
  { PyTorch ROCm dla Windows: RDNA3/RDNA4 oraz Ryzen AI Max / AI 300 (jak w bootstrap.ps1) }
  L := Lowercase(Names);
  Result := (Pos('radeon rx 7', L) > 0) or (Pos('radeon(tm) rx 7', L) > 0) or
            (Pos('radeon rx 9', L) > 0) or (Pos('radeon(tm) rx 9', L) > 0) or
            (Pos('radeon ai pro', L) > 0) or (Pos('radeon pro w7', L) > 0) or
            (Pos('8060s', L) > 0) or (Pos('8050s', L) > 0) or
            (Pos('890m', L) > 0) or (Pos('880m', L) > 0);
end;

function AutoVariant(const Names: String): Integer;
begin
  if Pos('nvidia', Lowercase(Names)) > 0 then
    Result := VariantNvidia
  else if IsSupportedAmd(Names) then
    Result := VariantAmd
  else
    Result := VariantCpu;
end;

function VariantName(Index: Integer): String;
begin
  case Index of
    VariantNvidia: Result := 'nvidia';
    VariantAmd: Result := 'amd';
  else
    Result := 'cpu';
  end;
end;

function VariantFromName(const Name: String): Integer;
begin
  if Name = 'nvidia' then
    Result := VariantNvidia
  else if Name = 'amd' then
    Result := VariantAmd
  else if Name = 'cpu' then
    Result := VariantCpu
  else
    Result := -1;
end;

function VariantLabel(Index: Integer): String;
begin
  case Index of
    VariantNvidia: Result := 'NVIDIA (CUDA)';
    VariantAmd: Result := 'AMD (ROCm, eksperymentalnie)';
  else
    Result := 'Tylko procesor (CPU)';
  end;
end;

function NeededGb(Index: Integer): Integer;
begin
  case Index of
    VariantNvidia: Result := 12;
    VariantAmd: Result := 10;
  else
    Result := 5;
  end;
end;

function DownloadEstimate(Index: Integer): String;
begin
  case Index of
    VariantNvidia: Result := 'ok. 4 GB';
    VariantAmd: Result := 'ok. 3 GB';
  else
    Result := 'ok. 1 GB';
  end;
end;

procedure InitializeWizard();
var
  Selected: Integer;
  Detected: String;
begin
  DetectedGpus := GetVideoControllers();
  if DetectedGpus = '' then
    Detected := 'nie wykryto'
  else
    Detected := DetectedGpus;

  GpuPage := CreateInputOptionPage(wpSelectDir,
    'Karta graficzna',
    'Na czym LinguaBee ma liczyć transkrypcję i tłumaczenie?',
    'Wykryte karty: ' + Detected + #13#10#13#10 +
    'Instalator pobierze PyTorch dopasowany do wybranej opcji. Wybór można zmienić, ' +
    'uruchamiając instalator ponownie.',
    True, False);
  GpuPage.Add('NVIDIA (CUDA) - karty GeForce / RTX; pobieranie ok. 4 GB');
  GpuPage.Add('AMD (ROCm) - Radeon RX 7000/9000, Ryzen AI Max; eksperymentalnie, sterownik Adrenalin 26.2.2+');
  GpuPage.Add('Tylko procesor (CPU) - działa na każdym komputerze, wolniej; pobieranie ok. 1 GB');

  Selected := VariantFromName(Lowercase(ExpandConstant('{param:GPU|}')));
  if Selected < 0 then
    Selected := VariantFromName(GetPreviousData('Gpu', ''));
  if Selected < 0 then
    Selected := AutoVariant(DetectedGpus);
  GpuPage.SelectedValueIndex := Selected;
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'Gpu', VariantName(GpuPage.SelectedValueIndex));
end;

{ ---------------------------------------------------------------- walidacja folderu }

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir, L: String;
  FreeBytes, TotalBytes: Int64;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Dir := RemoveBackslashUnlessRoot(WizardDirValue());
    L := Lowercase(Dir);
    if (Pos(Lowercase(ExpandConstant('{commonpf64}')), L) = 1) or
       (Pos(Lowercase(ExpandConstant('{commonpf32}')), L) = 1) then
    begin
      SuppressibleMsgBox('LinguaBee zapisuje modele i wyniki w folderze instalacji, więc nie może być ' +
        'zainstalowany w Program Files. Wybierz np. C:\LinguaBee albo D:\LinguaBee.', mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    if Pos('onedrive', L) > 0 then
    begin
      SuppressibleMsgBox('Folder w OneDrive nie nadaje się na instalację (kilka GB synchronizowanych plików). ' +
        'Wybierz np. C:\LinguaBee albo D:\LinguaBee.', mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    if FileExists(AddBackslash(Dir) + 'main.py') and not FileExists(AddBackslash(Dir) + 'install.json') then
    begin
      SuppressibleMsgBox('W tym folderze jest kod źródłowy LinguaBee (repozytorium). Wybierz inny folder, ' +
        'np. ' + Dir + '-App albo D:\LinguaBee.', mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    if Length(Dir) > 60 then
      if SuppressibleMsgBox('Ścieżka ma ' + IntToStr(Length(Dir)) + ' znaków. Modele i biblioteki mają bardzo ' +
        'długie ścieżki plików, co może powodować błędy. Zalecany jest krótki folder, np. D:\LinguaBee.' + #13#10#13#10 +
        'Kontynuować mimo to?', mbConfirmation, MB_YESNO, IDYES) = IDNO then
      begin
        Result := False;
        Exit;
      end;
  end;

  // Aktualizacja istniejącej instalacji pobiera niewiele - kontrola miejsca tylko przy nowej.
  if (CurPageID = GpuPage.ID) and not FileExists(AddBackslash(WizardDirValue()) + 'python\python.exe') then
  begin
    if GetSpaceOnDisk64(ExtractFileDrive(WizardDirValue()), FreeBytes, TotalBytes) then
      if FreeBytes < Int64(NeededGb(GpuPage.SelectedValueIndex)) * 1024 * 1024 * 1024 then
        if SuppressibleMsgBox(Format('Na dysku %s jest %d GB wolnego miejsca, a potrzeba ok. %d GB ' +
          '(bez modeli, które pobierają się później). Kontynuować mimo to?', [
          ExtractFileDrive(WizardDirValue()), Integer(FreeBytes div (1024 * 1024 * 1024)),
          NeededGb(GpuPage.SelectedValueIndex)]), mbConfirmation, MB_YESNO, IDNO) = IDNO then
          Result := False;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
    'Karta graficzna:' + NewLine + Space + VariantLabel(GpuPage.SelectedValueIndex) + NewLine;
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + MemoTasksInfo + NewLine;
  Result := Result + NewLine +
    'Z internetu zostaną pobrane: Python, PyTorch, biblioteki i ffmpeg (' +
    DownloadEstimate(GpuPage.SelectedValueIndex) + '). Modele (np. Whisper) pobierają się ' +
    'przy pierwszym użyciu.';
end;

{ ---------------------------------------------------------------- program uruchomiony? }

function IsRunningFrom(const AppDir: String): Boolean;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  Prefix, Path: String;
begin
  Result := False;
  Prefix := Lowercase(AddBackslash(AppDir) + 'python\');
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT ExecutablePath FROM Win32_Process WHERE Name = ''python.exe'' OR Name = ''pythonw.exe''');
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      if not VarIsNull(Item.ExecutablePath) then
      begin
        Path := Item.ExecutablePath;
        if Pos(Prefix, Lowercase(Path)) = 1 then
        begin
          Result := True;
          Exit;
        end;
      end;
    end;
  except
    Result := False;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if IsRunningFrom(WizardDirValue()) then
    Result := 'LinguaBee jest uruchomiony z tego folderu. Zamknij program, a potem kliknij „Wstecz” i „Dalej”.';
end;

{ ---------------------------------------------------------------- pobieranie składników }

function StepLabel(Step: Integer): String;
begin
  case Step of
    1: Result := 'Sprawdzanie systemu...';
    2: Result := 'Wybór wariantu dla karty graficznej...';
    3: Result := 'Pobieranie Pythona...';
    4: Result := 'Pobieranie PyTorch i bibliotek (kilka GB, może potrwać kilkanaście minut)...';
    5: Result := 'Pobieranie ffmpeg...';
    6: Result := 'Pobieranie Deno (YouTube)...';
    7: Result := 'Test programu...';
  else
    Result := 'Instalowanie składników...';
  end;
end;

procedure BootstrapOutput(const S: String; const Error, FirstLine: Boolean);
var
  P, Step: Integer;
  Rest, Line: String;
begin
  Line := Trim(S);
  P := Pos('@@KROK ', Line);
  if P > 0 then
  begin
    Rest := Copy(Line, P + 7, Length(Line));
    Step := StrToIntDef(Copy(Rest, 1, Pos('/', Rest) - 1), 0);
    if (Step >= 1) and (Step <= BootstrapSteps) then
    begin
      WizardForm.StatusLabel.Caption := StepLabel(Step);
      WizardForm.ProgressGauge.Position := Step - 1;
    end;
  end
  else if Line <> '' then
    WizardForm.FilenameLabel.Caption := Copy(Line, 1, 110);
end;

procedure RunBootstrap();
var
  Params: String;
begin
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' +
    ExpandConstant('{app}\app\packaging\installer\bootstrap.ps1') + '" -Root "' + ExpandConstant('{app}') +
    '" -Gpu ' + VariantName(GpuPage.SelectedValueIndex);
  if not WizardIsTaskSelected('youtube') then
    Params := Params + ' -NoDeno';

  WizardForm.ProgressGauge.Style := npbstNormal;
  WizardForm.ProgressGauge.Min := 0;
  WizardForm.ProgressGauge.Max := BootstrapSteps;
  WizardForm.ProgressGauge.Position := 0;
  WizardForm.StatusLabel.Caption := StepLabel(0);
  WizardForm.FilenameLabel.Caption := '';
  Log('Bootstrap: powershell.exe ' + Params);
  if not ExecAndLogOutput(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params,
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, BootstrapResult, @BootstrapOutput) then
    BootstrapResult := -1;
  WizardForm.ProgressGauge.Position := BootstrapSteps;
  Log('Bootstrap: kod wyjścia ' + IntToStr(BootstrapResult));

  if BootstrapResult = 2 then
    SuppressibleMsgBox('LinguaBee jest zainstalowany, ale niektóre składniki zgłosiły ostrzeżenia ' +
      '(np. ffmpeg albo Deno). Szczegóły: ' + ExpandConstant('{app}\user\logs\install.log') + #13#10#13#10 +
      'Możesz to poprawić później: menu Start > LinguaBee > Napraw instalację.', mbInformation, MB_OK, IDOK)
  else if BootstrapResult <> 0 then
    SuppressibleMsgBox('Nie udało się pobrać wszystkich składników LinguaBee (kod ' + IntToStr(BootstrapResult) + ').' + #13#10#13#10 +
      'Szczegóły: ' + ExpandConstant('{app}\user\logs\install.log') + #13#10#13#10 +
      'Sprawdź połączenie z internetem i uruchom: menu Start > LinguaBee > Napraw instalację.',
      mbError, MB_OK, IDOK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    RunBootstrap();
end;

function BootstrapSucceeded(): Boolean;
begin
  Result := (BootstrapResult = 0) or (BootstrapResult = 2);
end;

function GetCustomSetupExitCode(): Integer;
begin
  { Instalacja cicha: 101 = pliki skopiowane, ale pobieranie składników się nie udało. }
  if BootstrapSucceeded() then
    Result := 0
  else
    Result := 101;
end;

{ ---------------------------------------------------------------- deinstalacja }

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    DeleteUserData := SuppressibleMsgBox('Usunąć także modele, wyniki i ustawienia LinguaBee ' +
      '(foldery models, output i user w ' + ExpandConstant('{app}') + ')?' + #13#10#13#10 +
      'Wybierz „Nie”, żeby je zachować - przydadzą się po ponownej instalacji.',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES;
  if (CurUninstallStep = usPostUninstall) and DeleteUserData then
  begin
    DelTree(ExpandConstant('{app}\models'), True, True, True);
    DelTree(ExpandConstant('{app}\output'), True, True, True);
    DelTree(ExpandConstant('{app}\user'), True, True, True);
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
