# Windows packaging and Microsoft Store distribution

How the native Windows build is produced, how the installer is generated,
what code signing the Store requires, and the exact submission shape a
VocaGateway listing would take. The packaging inputs live in
`packaging/windows/`; the pipeline lives in
`.github/workflows/windows-package.yml`.

## What the pipeline produces

1. **`dist/vocagateway/`** — a PyInstaller one-directory build. `vocagateway.exe`
   is `app.cli.serve` frozen with CPython 3.12, the HTMX WebUI, Jinja2
   templates, the model pin files, and every engine that has a `win_amd64`
   wheel (sherpa-onnx, faster-whisper / ctranslate2, moonshine-voice) bundled
   next to it. The spec also drops a `ffmpeg.exe` beside the app binary —
   `app.audio` resolves it via `sys.executable` when `sys.frozen` is set, so
   the install is self-contained.
2. **`VocaGatewaySetup-<version>.exe`** — an Inno Setup installer. Per-user
   install to `%LOCALAPPDATA%\Programs\VocaGateway` with
   `PrivilegesRequired=lowest`, so **no UAC prompt, no admin rights needed**.
   It adds Start Menu entries (run the server, open the WebUI, uninstall) and
   an opt-in "start on sign-in" `HKCU\Run` task.

## Building locally or in CI

```powershell
uv sync --locked --no-dev --extra engines
uv run --with pyinstaller pyinstaller packaging/windows/vocagateway.spec --noconfirm --clean
# drop ffmpeg.exe into dist\vocagateway\ (see the workflow's "Bundle ffmpeg.exe" step)
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\setup.iss /DAppVersion=0.1.0
```

CI does all of this on `windows-latest` on every relevant PR (build-only) and
attaches the installer to published GitHub releases — a release asset URL is
exactly the **versioned HTTPS URL** a Store MSI/EXE submission requires
(`https://github.com/VocaHQ/vocagateway/releases/download/vX.Y.Z/VocaGatewaySetup-X.Y.Z.exe`).

## EXE vs MSI

The Store's unpackaged listing accepts either an `.exe` or an `.msi`. We ship
an Inno Setup EXE: MSI generation means a WiX Toolset project authoring a
component-per-file manifest — more machinery, no benefit for a per-user app
that wants no system integration. If MSI is ever wanted, the WiX `wixl` path
exists, but the EXE satisfies every stated requirement.

## What the Store MSI/EXE listing needs

Per Microsoft's package requirements:

- Standalone offline installer — ours is (no downloaders/stubs).
- Silent install — Inno Setup: `VocaGatewaySetup-x.y.z.exe /VERYSILENT /NORESTART /SP-`.
  CI smoke-tests exactly this invocation.
- **Authenticode signature on the installer *and every PE inside it***
  (exe + all DLLs/pyd files), chaining to a Microsoft Trusted Root CA. The
  Store does not re-sign MSI/EXE submissions — this is the one real cost.
  Options:
  - Azure Trusted Signing (~$9.99/mo) — cheapest, cloud HSM, `signtool` works
    via the Trusted Signing dlib in CI.
  - Standard OV code-signing cert (~$100–400/yr) from a public CA.
  - Signing step is intentionally absent from the workflow until a
    certificate exists; adding it is one `signtool sign` invocation per PE
    (or `--signtool` in PyInstaller for the exe plus Inno's `SignTool`
    directive for the installer wrapper).
- A versioned URL whose binary never changes — GitHub Releases assets are
  immutable once uploaded, which is what the `/releases/download/<tag>/`
  URL gives us.

## Submission checklist (Partner Center side)

Human steps, none of which Devin can do for you:

1. Enroll a Partner Center developer account — Individual is **$0** under the
   new onboarding flow (ID + selfie verification; Company needs business
   verification).
2. Reserve the app name (`New product → EXE or MSI app`).
3. Fill the submission: availability/pricing, properties, IARC age-rating
   questionnaire (VocaGateway rates clean), Packages page (the release-asset
   URL + `/VERYSILENT /NORESTART /SP-` installer parameters, arch x64,
   type EXE), Store listing text/screenshots, then **Submit for
   certification** — up to 3 business days.

Automation alternative: the Microsoft Store submission API can drive all of
step 3 from CI, but it needs an Azure AD app registration on *your* Partner
Center account — optional, worth it only if releases become frequent.

## Known gaps in the current build

- **Console window**: `vocagateway.exe` is a console app, so the
  start-on-login task opens a terminal window with the server log. A tray
  icon or session-0 service mode is the natural follow-up (MSIX packaged
  services also exist if we ever go the MSIX route instead).
- **Code signing**: unsigned until a certificate is provisioned — fine for
  sideloading with a SmartScreen warning, required before Store submission.
- **arm64**: the installer is x64-only (`ArchitecturesAllowed=x64compatible`);
  WoW covers Snapdragon machines.
- **LAN firewall prompt**: the gateway binds `0.0.0.0:8765` for phone pairing,
  so first run triggers the standard Windows Defender Firewall allow dialog —
  expected, private-network scope.
- **Config paths stay XDG-style**: `%USERPROFILE%\.config\vocagateway` rather
  than `%LOCALAPPDATA%` — functional, just not idiomatic Windows.
