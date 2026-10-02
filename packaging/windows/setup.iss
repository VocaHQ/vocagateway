; Inno Setup script for the VocaGateway Windows installer.
;
; Produces a single VocaGatewaySetup-<version>.exe that installs per-user into
; %LOCALAPPDATA%\Programs\VocaGateway — no administrator rights, no elevation
; prompts, so the package stays eligible for a Microsoft Store MSI/EXE listing
; (silent install: VocaGatewaySetup.exe /VERYSILENT /NORESTART /SP-).
;
; Build:
;   uv run --with pyinstaller pyinstaller packaging/windows/vocagateway.spec
;   <drop ffmpeg.exe into dist\vocagateway\>
;   & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\setup.iss /DAppVersion=0.1.0

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "VocaGateway"
#define AppPublisher "VocaHQ"
#define AppURL "https://github.com/VocaHQ/vocagateway"
#define AppExeName "vocagateway.exe"

[Setup]
AppId={{8F4E2B1A-9C3D-4A5E-B7F6-2E1C8D4A5B96}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\Programs\VocaGateway
DefaultGroupName=VocaGateway
PrivilegesRequired=lowest
OutputDir=..\..\Output
OutputBaseFilename=VocaGatewaySetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
; Closing the gateway before install is best-effort; it may not be running.
CloseApplications=force

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startup"; Description: "Start VocaGateway when you sign in"; GroupDescription: "Startup:"

[Files]
Source: "..\..\dist\vocagateway\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VocaGateway"; Filename: "{app}\{#AppExeName}"; Comment: "Run the VocaGateway transcription server"
Name: "{group}\VocaGateway WebUI"; Filename: "http://localhost:8765/"; Comment: "Open the VocaGateway admin UI"
Name: "{group}\Uninstall VocaGateway"; Filename: "{uninstallexe}"

[Registry]
; The startup task is opt-in at install time. HKCU\Run needs no elevation.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VocaGateway"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start VocaGateway now"; Flags: nowait postinstall skipifsilent unchecked
