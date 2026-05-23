#!/usr/bin/env bash
# setup.sh — Git Bash installer for face-mask.
#
# Detects whether the active Python is the Microsoft Store "execution alias"
# stub (which breaks Spout's discrete-GPU selection), installs python.org
# Python via winget if needed, recreates the .venv against a real interpreter,
# and installs requirements.
#
# Idempotent — safe to run multiple times.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PY_VERSION="3.13"
WINGET_PKG="Python.Python.$PY_VERSION"

# --- helpers ----------------------------------------------------------------

# Resolve to a Windows path so .exe invocations work cleanly from Git Bash.
to_winpath() {
    cygpath -w "$1" 2>/dev/null || echo "$1"
}

is_store_alias() {
    # Microsoft Store "execution aliases" are 0-byte reparse points under
    # %LOCALAPPDATA%\Microsoft\WindowsApps. We identify them by EITHER
    # living under that path OR being zero bytes (genuine python.exe binaries
    # are small but never zero).
    local exe="$1"
    [[ "$exe" == *"WindowsApps"* ]] && return 0
    [[ -e "$exe" && ! -s "$exe" ]] && return 0  # exists but zero bytes
    return 1
}

# Accept this path as a real python.org interpreter if it's not a Store alias
# AND it ships with a pythonXY.dll sibling (the actual interpreter — required
# because modern python.exe is just a ~100 KB loader stub).
is_real_python() {
    local p="$1"
    [[ -z "$p" ]] && return 1
    [[ ! -f "$p" ]] && return 1
    is_store_alias "$p" && return 1

    # Must be non-zero. The loader stub itself is small (~100 KB on 3.10+),
    # so we no longer gate on a 1 MB threshold.
    local size
    size=$(stat -c%s "$p" 2>/dev/null || echo 0)
    [[ "$size" -lt 50000 ]] && return 1  # smaller than even the loader stub

    # Require a sibling pythonXY.dll. Real installs always have one; bogus
    # python.exe binaries from unrelated tools (TouchDesigner, LibreOffice
    # embedded interpreters, build artifacts) don't have a top-level pythonXY.dll
    # in the same dir AND wouldn't pass detection anyway.
    local dir
    dir="$(dirname "$p")"
    local has_pydll=0
    for dll in "$dir"/python3*.dll; do
        if [[ -f "$dll" ]]; then
            has_pydll=1
            break
        fi
    done
    [[ "$has_pydll" -eq 1 ]]
}

# Locate py.exe (the Python Launcher). Per-user installs land it at
# %LOCALAPPDATA%\Programs\Python\Launcher\py.exe; all-users installs land it
# at C:\Windows\py.exe. PATH may not reflect a just-installed copy in this
# shell, so we probe absolute paths.
locate_py_launcher() {
    local localappdata_unix
    localappdata_unix="$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || echo "$LOCALAPPDATA")"

    local candidates=(
        "$localappdata_unix/Programs/Python/Launcher/py.exe"
        "/c/Windows/py.exe"
        "/c/Windows/System32/py.exe"
    )
    for path in "${candidates[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    command -v py.exe 2>/dev/null || return 1
}

# Ask the Python Launcher to enumerate every Python install it knows about.
# `py -0p` prints something like:
#     -V:3.13 *        C:\Users\cdelg\AppData\Local\Programs\Python\Python313\python.exe
#     -V:3.12          C:\...\python.exe
# This is the most authoritative source — the launcher reads its config from
# %LOCALAPPDATA%\py.ini AND from the registry, so it sees freshly-installed
# python.org Pythons even before PATH refreshes.
find_via_py_launcher() {
    local launcher
    launcher="$(locate_py_launcher || true)"
    [[ -z "$launcher" ]] && return 1

    # Capture the listing
    local listing
    listing="$("$launcher" -0p 2>/dev/null || true)"
    [[ -z "$listing" ]] && return 1

    # Pick the highest-version real python from the listing
    while IFS= read -r line; do
        # The path is the last whitespace-delimited token on the line
        local exe_win
        exe_win="$(echo "$line" | awk '{print $NF}')"
        [[ "$exe_win" != *python.exe ]] && continue

        local exe_unix
        exe_unix="$(cygpath -u "$exe_win" 2>/dev/null || echo "$exe_win")"
        if is_real_python "$exe_unix"; then
            echo "$exe_unix"
            return 0
        fi
    done <<< "$listing"
    return 1
}

# Parse `winget list` to find an install directory for Python.Python.3.x.
# This rescues the case where winget installed Python somewhere outside our
# usual search paths.
find_via_winget_list() {
    command -v winget >/dev/null 2>&1 || return 1

    # `winget list` prints columns; we look for our package id and pull the
    # install location from `winget show`. `--exact` keeps fuzzy matches out.
    for ver in 3.13 3.12 3.11; do
        local pkg="Python.Python.$ver"
        if winget list --id "$pkg" --exact 2>/dev/null | grep -q "$pkg"; then
            # Extract the install location, if reported.
            local loc_win
            loc_win="$(winget show --id "$pkg" --exact 2>/dev/null \
                | awk -F': ' '/Install location/ {print $2; exit}')"
            if [[ -n "$loc_win" ]]; then
                local loc_unix
                loc_unix="$(cygpath -u "$loc_win" 2>/dev/null || echo "$loc_win")"
                local candidate="$loc_unix/python.exe"
                if is_real_python "$candidate"; then
                    echo "$candidate"
                    return 0
                fi
            fi
        fi
    done
    return 1
}

# Search standard install locations for any python.org interpreter.
# Globs each parent directory so we catch versioned/architecture-suffixed
# subdirectories (e.g. Python313, Python313-arm64, Python312, ...).
find_in_standard_locations() {
    # Convert $LOCALAPPDATA (Windows path) to a Unix path Git Bash can glob.
    local localappdata_unix
    localappdata_unix="$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || echo "$LOCALAPPDATA")"

    local roots=(
        "$localappdata_unix/Programs/Python"
        "/c/Program Files/Python"  # python.org system-wide installs go to /c/Program Files/PythonXYZ
        "/c"                        # /c/Python313 etc.
        "/c/Program Files"          # /c/Program Files/Python313 etc.
    )

    for root in "${roots[@]}"; do
        [[ -d "$root" ]] || continue
        # Match Python*, Python3*, Python313, Python313-arm64, ...
        for subdir in "$root"/Python* "$root"/python*; do
            [[ -d "$subdir" ]] || continue
            local candidate="$subdir/python.exe"
            if is_real_python "$candidate"; then
                echo "$candidate"
                return 0
            fi
        done
    done
    return 1
}

# Combined detector — tries the cheapest, most reliable signals first.
# Each finder prints to stdout on success; stderr is used for diagnostics so
# the chosen path is the only stdout content the caller captures.
find_python_org() {
    local result

    result="$(find_in_standard_locations || true)"
    if [[ -n "$result" ]]; then
        echo "    [hit] standard location: $result" >&2
        echo "$result"
        return 0
    fi
    echo "    [miss] no python.exe under standard install roots" >&2

    result="$(find_via_py_launcher || true)"
    if [[ -n "$result" ]]; then
        echo "    [hit] py.exe launcher: $result" >&2
        echo "$result"
        return 0
    fi
    echo "    [miss] py.exe launcher reported no real installs" >&2

    result="$(find_via_winget_list || true)"
    if [[ -n "$result" ]]; then
        echo "    [hit] winget list/show: $result" >&2
        echo "$result"
        return 0
    fi
    echo "    [miss] winget list/show returned no usable install path" >&2

    return 1
}

install_python_winget() {
    echo "==> Installing $WINGET_PKG via winget..."
    if ! command -v winget >/dev/null 2>&1; then
        echo "    winget not available — skipping to direct download."
        return 1
    fi

    # --force: install even if winget incorrectly thinks the package is present
    #          (happens when Store Python registers similarly in ARP)
    # --source winget: pin to the curated source, ignore Store entries
    winget install --id "$WINGET_PKG" --exact \
        --source winget \
        --scope user --silent --force \
        --accept-source-agreements --accept-package-agreements \
        || return 1
    return 0
}

# Direct download from python.org. Used when winget either fails or silently
# no-ops (which it does when Defender/AppLocker/an existing ARP entry causes
# the installer to exit 0 without writing files).
install_python_direct() {
    local installer_url="https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe"
    local tmp_installer="/tmp/python-3.13.1-amd64.exe"

    if [[ ! -s "$tmp_installer" ]]; then
        echo "==> Downloading $installer_url"
        if ! command -v curl >/dev/null 2>&1; then
            echo "ERROR: curl not available for fallback download."
            return 1
        fi
        curl -L -o "$tmp_installer" "$installer_url" || return 1
    else
        echo "==> Reusing cached installer at $tmp_installer"
    fi

    echo "==> Running python.org installer"
    echo "    A Windows installer dialog will open. Recommended choices:"
    echo "      - Check 'Add python.exe to PATH'"
    echo "      - Click 'Install Now'"
    echo "      - Wait for 'Setup was successful', then close the dialog."
    echo

    # Launch with full UI so the user can see and react to any failure.
    # `start /wait` blocks this script until the installer dialog is closed.
    local installer_win
    installer_win="$(cygpath -w "$tmp_installer")"
    cmd //c "start /wait \"\" \"$installer_win\"" || {
        echo "Failed to launch installer at $installer_win"
        return 1
    }

    rm -f "$tmp_installer"
    return 0
}

# --- main -------------------------------------------------------------------

echo "==> face-mask setup"
echo "    project: $PROJECT_DIR"

# 1. Inspect current python on PATH (informational only).
if command -v python >/dev/null 2>&1; then
    CURRENT_PY="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [[ -n "${CURRENT_PY:-}" ]]; then
        echo "    current PATH python: $CURRENT_PY"
        if is_store_alias "$(cygpath -u "$CURRENT_PY")"; then
            echo "    -> this is a Microsoft Store stub (will be replaced)"
        fi
    fi
fi

# 2. Locate (or install) a real python.org interpreter.
echo
echo "==> Locating a python.org Python $PY_VERSION (skipping Store stubs)..."
REAL_PY="$(find_python_org || true)"

if [[ -z "$REAL_PY" ]]; then
    # Attempt 1: winget. May silently no-op even on exit 0 (Defender / AppLocker
    # / stale ARP entry). We always re-verify, never trust winget's exit code.
    echo
    echo "==> No real Python found — attempting winget install"
    install_python_winget || true
    REAL_PY="$(find_python_org || true)"
fi

if [[ -z "$REAL_PY" ]]; then
    # Attempt 2: direct download from python.org with the visible installer UI.
    # If winget's silent install secretly failed, the visible installer will
    # surface the actual error (e.g. "Setup failed: 0x80070643") so the user
    # can react.
    echo
    echo "==> winget did not produce a working install — falling back to direct download"
    install_python_direct || true
    REAL_PY="$(find_python_org || true)"
fi

if [[ -z "$REAL_PY" ]]; then
    cat <<EOF

ERROR: Could not locate a real python.org Python after every install attempt.

Diagnostics to paste back so we can figure out where Python is (or why install
keeps failing):

  ls "/c/Users/\$USER/AppData/Local/Programs/Python/" 2>/dev/null
  ls "/c/Program Files/" 2>/dev/null | grep -i python
  /c/Windows/py.exe -0p 2>/dev/null
  winget list Python.Python.3.13 --exact 2>/dev/null

If the python.org installer dialog appeared but reported an error, the message
on its final page is what we need (often something like 0x80070643, which
indicates a previous install left a broken MSI registration).

Manual install always works: https://www.python.org/downloads/
After install, re-run: bash setup.sh
EOF
    exit 1
fi
echo "    using: $REAL_PY"

# 3. Sanity check: defer to is_real_python (which already enforces non-Store,
#    non-zero, and presence of a pythonXY.dll sibling — the right test for
#    modern python.org installs where python.exe is a ~100 KB loader stub).
REAL_PY_SIZE=$(stat -c%s "$REAL_PY")
echo "    size:  $((REAL_PY_SIZE / 1024)) KB"
if ! is_real_python "$REAL_PY"; then
    echo "ERROR: located interpreter is not a real python.org install."
    exit 1
fi

# 4. Recreate the venv from scratch so it's bound to the new interpreter.
if [[ -d "$VENV_DIR" ]]; then
    echo
    echo "==> Removing existing .venv..."
    rm -rf "$VENV_DIR"
fi

echo "==> Creating .venv with $REAL_PY"
"$REAL_PY" -m venv "$VENV_DIR"

# 5. Install dependencies into the new venv.
VENV_PY="$VENV_DIR/Scripts/python.exe"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv creation did not produce $VENV_PY"
    exit 1
fi

echo
echo "==> Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
    echo
    echo "==> Installing requirements.txt"
    "$VENV_PY" -m pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "WARNING: no requirements.txt found at $PROJECT_DIR/requirements.txt"
fi

# 6. Print the interpreter graph so the user can see what the venv resolves to.
echo
echo "==> Verifying venv interpreter chain"
"$VENV_PY" -c "
import sys, os
print(f'  sys.executable        = {sys.executable}')
print(f'  sys._base_executable  = {getattr(sys, \"_base_executable\", None)}')
print(f'  base size             = {os.path.getsize(sys._base_executable)} bytes')
"

# 7. Pre-register the high-performance GPU preference so the first real run
#    does not need to relaunch.
echo
echo "==> Registering GPU preference (HKCU\\...\\UserGpuPreferences)"
# Run from inside PROJECT_DIR so Python's `-c` mode finds gpu_preference.py
# via its implicit cwd-on-sys.path. No shell-to-Python string interpolation
# (which previously injected literal backslashes into a Python string,
# triggering "SyntaxWarning: invalid escape sequence '\c'").
(
    cd "$PROJECT_DIR" && "$VENV_PY" -c "
from gpu_preference import ensure_high_performance_gpu
ensure_high_performance_gpu(auto_relaunch=False)
"
) || echo "(skipping — gpu_preference.py not yet imported successfully)"

cat <<EOF

==> Setup complete.

To run:
  source .venv/Scripts/activate
  python main.py

On first launch you should see Spout report:
  [Spout]   GL_RENDERER: NVIDIA GeForce ...

If GL_RENDERER still says Intel, the registry preference was overridden by
NVIDIA Control Panel. Open NVIDIA CP -> Manage 3D Settings -> Program Settings
and add this exe with "High-performance NVIDIA processor":
  $(to_winpath "$VENV_PY")
EOF
