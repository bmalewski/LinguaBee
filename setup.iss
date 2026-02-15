[Setup]
AppName=LinguaBee
AppVersion=1.0
AppPublisher=Benedykt Malewski
DefaultDirName={autopf}\LinguaBee
DefaultGroupName=LinguaBee
OutputDir=dist_installer
OutputBaseFilename=LinguaBeeInstaller
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
UninstallDisplayIcon={app}\LinguaBee.exe
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"

[Files]
; Jeśli masz jeden plik .exe:
Source: "dist\LinguaBee.exe"; DestDir: "{app}"; Flags: ignoreversion
; Jeśli masz folder z wieloma plikami, użyj tego zamiast powyższego:
; Source: "dist\LinguaBee\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\LinguaBee"; Filename: "{app}\LinguaBee.exe"
Name: "{autodesktop}\LinguaBee"; Filename: "{app}\LinguaBee.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Utwórz ikonę na pulpicie"; GroupDescription: "Dodatkowe skróty:"

[Run]
Filename: "{app}\LinguaBee.exe"; Description: "Uruchom LinguaBee"; Flags: nowait postinstall skipifsilent
