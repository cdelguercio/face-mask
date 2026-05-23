# Installation

Quick-start installer for **face-mask** on Windows + Git Bash. Handles the most common landmine: Microsoft Store Python, which silently breaks Spout's discrete-GPU selection and causes the receiver to show a 0×0 texture.

## TL;DR

```bash
cd /d/claude/face-mask
bash setup.sh
source .venv/Scripts/activate
python main.py
```

If `setup.sh` reports "Setup complete" and `main.py` prints `[Spout] GL_RENDERER: NVIDIA GeForce ...`, you're done.

## Requirements

- **Windows 10 1903+ or Windows 11** (for the per-app GPU preference registry).
- **Git Bash** ([git-scm.com](https://git-scm.com/download/win)).
- **winget** (preinstalled on Windows 10/11; required only if Python isn't already installed).
- **NVIDIA GPU** for Spout to share textures with Resolume on the discrete adapter. Integrated-only systems will not work for projection — Spout requires GPU-to-GPU sharing on the same adapter.
- **Resolume Arena/Avenue** as the Spout receiver.

## What `setup.sh` does

1. **Inspects your current Python** and flags it if it's the Microsoft Store version. Store Python ships as a zero-byte execution alias under `WindowsApps\` and routes through a UWP sandbox — it breaks GPU adapter selection, native extensions, and other things this project relies on.
2. **Finds a real python.org Python 3.13** in any of the standard locations:
   - `%LOCALAPPDATA%\Programs\Python\Python313\`
   - `C:\Python313\`
   - `C:\Program Files\Python313\`

   It also accepts Python 3.12 and 3.11 if 3.13 is absent. Anything under `WindowsApps\` or under 1 MB is rejected as a stub.
3. **If no real Python is found, installs it via winget:**
   ```
   winget install --id Python.Python.3.13 --scope user --silent
   ```
   No admin prompt, no installer UI. If winget isn't available, the script tells you to install Python manually from [python.org/downloads](https://www.python.org/downloads/) and re-run.
4. **Recreates `.venv` from scratch** against the real interpreter. Any pre-existing `.venv` is deleted — necessary because a venv is bound to its base interpreter at creation time, and we need the new venv to point at the new Python.
5. **Installs `requirements.txt`** into the fresh venv.
6. **Verifies the interpreter chain** by printing `sys.executable`, `sys._base_executable`, and the base binary size. The base must be a multi-MB real binary, not a 0-byte alias.
7. **Pre-registers the high-performance GPU preference** under `HKCU\SOFTWARE\Microsoft\DirectX\UserGpuPreferences` for every python.exe involved, so the first `python main.py` doesn't need to self-relaunch.

The script is **idempotent** — safe to run as many times as you want. If everything is already correct, it just verifies and exits.

## Run

```bash
cd /d/claude/face-mask
bash setup.sh                       # one-time, ~3 minutes if winget runs
source .venv/Scripts/activate
python main.py
```

On the first `python main.py`, watch for:

```
[Spout] pygame 2.6.x, SpoutGL 0.0.x
[Spout]   GL_VENDOR  : NVIDIA Corporation
[Spout]   GL_RENDERER: NVIDIA GeForce RTX 4060
[Spout] Sender 'FaceMaskSpout' registered at 1920x1080.
```

If `GL_RENDERER` says NVIDIA, Spout will work. Add the source in Resolume and you should see the live mask.

## Troubleshooting

### `GL_RENDERER` still says Intel after running `setup.sh`

The Windows registry preference is set but NVIDIA's own profile is overriding it. Open **NVIDIA Control Panel → Manage 3D Settings → Program Settings**, add the venv's python.exe with "High-performance NVIDIA processor":

```
D:\claude\face-mask\.venv\Scripts\python.exe
```

The setup script prints this exact path at the end.

### Resolume shows the sender name but a 0×0 texture

Three checks, in order:

1. **`GL_RENDERER` line** — if it says Intel, see above.
2. **Reboot** — Windows occasionally loses Spout's shared-memory mappings after a driver crash. Always worth trying.
3. **Spout SDK runtime** — install or reinstall the [Spout 2.7 distribution](https://leadedge.github.io/). Its installer registers COM components that Windows Defender sometimes silently removes.

### Periodic `[Spout] X/Y frames sent OK, Z failed` warning

`sendImage` is returning false for some frames. Usually a transient driver hiccup. If `Z` keeps growing rather than stabilizing, the GL context was lost — restart the app.

### `setup.sh` says winget isn't available

Open `cmd` and run `winget --version`. If that also fails, update Windows via Microsoft Store → Library → Updates, or install **App Installer** from the Store. Worst case, install Python manually from [python.org/downloads](https://www.python.org/downloads/) and re-run `setup.sh` — it'll find the manual install.

### "winget reported success but no python.org install was found"

`PATH` in your current Git Bash session is stale. Close the window, open a fresh Git Bash, and re-run `bash setup.sh`. The script is idempotent, so this is safe.

### Anything ImportError-related on first run

The fresh venv may need MediaPipe's model file. See the optional steps in `README.md` — the only one that's truly required for the default detector mode is:

```bash
curl -L -o face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Why not just `pip install -r requirements.txt` like a normal Python project?

You can — and if you already have a non-Store Python, that's all that's needed. `setup.sh` exists to handle the *one* configuration that silently breaks Spout's GPU sharing: a venv whose base interpreter is Microsoft Store Python. That setup fails in a way that produces no Python error — the receiver just sees a 0×0 texture and you have no idea why. The script makes that misconfiguration impossible.
