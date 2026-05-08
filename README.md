# Knoxnet VMS Beta

Knoxnet VMS Beta is an early public technical beta of the Knoxnet video management system. It is intended for testers, demo content, and real-world feedback before a polished installer is released.

This beta is free for up to 4 cameras. Paid license, payment, installer, code signing, PyInstaller packaging, and auto-update flows are not included yet.

## Beta Warning

This is early technical beta software. Use it for testing, demos, and feedback only. Do not rely on this beta for production-critical security use yet.

## Clone

```bash
git clone https://github.com/PhillipAlexanderYoung/Knoxnet-VMS
cd Knoxnet-VMS
```

## Linux Install And Start

```bash
./install.sh
./run.sh
```

Notes:

- Python 3.10 or newer is recommended.
- The app creates a local `.venv` and installs Python dependencies from `requirements.txt`.
- MediaMTX is downloaded automatically on first server start if the binary is missing.

## Windows Install And Start

```powershell
git clone https://github.com/PhillipAlexanderYoung/Knoxnet-VMS
cd Knoxnet-VMS
install.bat
run.bat
```

PowerShell equivalents:

```powershell
.\install.ps1
.\run.ps1
```

If PowerShell blocks scripts, run the `.bat` wrappers or start PowerShell with execution policy bypass for this session.

## Add Cameras

1. Start the desktop app.
2. Use the first-run setup wizard or tray menu camera configuration.
3. Add RTSP/IP cameras manually or use discovery where supported.
4. The public beta allows up to 4 configured cameras.

Camera credentials stay local. Do not commit `.env`, `data/`, `cameras.json`, recordings, captures, or logs.

## Updates

On startup, Knoxnet VMS Beta checks the public Knoxnet update endpoint for the latest beta version. If a newer beta is available, the app shows a non-blocking message:

```text
Update available: version X. Download the latest beta from GitHub.
```

The beta does not auto-update.

Manual update process:

1. Stop the app.
2. Delete your local `Knoxnet-VMS` folder.
3. Clone the latest repo again.
4. Run the install and start commands again.

Linux:

```bash
git clone https://github.com/PhillipAlexanderYoung/Knoxnet-VMS
cd Knoxnet-VMS
./install.sh
./run.sh
```

Windows:

```powershell
git clone https://github.com/PhillipAlexanderYoung/Knoxnet-VMS
cd Knoxnet-VMS
install.bat
run.bat
```

## Report Bugs

Please report bugs through GitHub Issues: https://github.com/PhillipAlexanderYoung/Knoxnet-VMS/issues

Include:

- Operating system and version
- Python version
- Install/start command used
- Camera model and stream type
- Number of cameras configured
- What happened and what you expected
- Relevant logs or screenshots, with credentials removed

## Public Repo Safety

This public beta intentionally excludes private Cloudflare Worker code, internal deployment tooling, paid entitlement/payment logic, customer deployment scripts, signing assets, installers, and generated local data. See `PUBLIC_RELEASE_CHECKLIST.md` before publishing a new beta snapshot.
