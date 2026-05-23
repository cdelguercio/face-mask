"""Force python.exe to use the high-performance (discrete) GPU on Windows.

Spout shares GPU textures via DirectX shared handles. Those handles are only
valid on the adapter that created them, so if python.exe runs on the Intel
iGPU but Resolume runs on the NVIDIA dGPU, the receiver sees a 0x0 texture.

We write per-app GPU preferences to:
    HKCU\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences

This is the same registry the Windows "Graphics settings" UI writes. The
preference is evaluated at process launch, so on first run we set it and
re-exec.

Wrinkle: modern Python venvs on Windows ship python.exe as a thin redirector
stub that re-launches the base interpreter in a NEW process. Windows decides
GPU adapter per-process, so the venv stub's preference doesn't affect the
real interpreter. We register every interpreter we can find behind the stub.
"""

import os
import sys

REG_PATH = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
DESIRED_VALUE = "GpuPreference=2;"  # 0=auto, 1=power-saving, 2=high-performance


def _read_registry_preference(exe_path: str) -> str | None:
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_READ)
    except FileNotFoundError:
        return None
    try:
        try:
            val, _ = winreg.QueryValueEx(key, exe_path)
            return val
        except FileNotFoundError:
            return None
    finally:
        winreg.CloseKey(key)


def _write_registry_preference(exe_path: str) -> bool:
    """Set GpuPreference=2 for exe_path. Returns True if value changed."""
    import winreg
    try:
        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, REG_PATH, 0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
    except OSError as e:
        print(f"[GPU] Could not open registry key: {e}")
        return False
    try:
        current = _read_registry_preference(exe_path)
        if current == DESIRED_VALUE:
            return False
        winreg.SetValueEx(key, exe_path, 0, winreg.REG_SZ, DESIRED_VALUE)
        return True
    finally:
        winreg.CloseKey(key)


def _venv_base_python() -> str | None:
    """Find the base interpreter a venv redirects to via pyvenv.cfg."""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    # pyvenv.cfg is one level up from Scripts/
    for cfg_dir in (exe_dir, os.path.dirname(exe_dir)):
        cfg_path = os.path.join(cfg_dir, "pyvenv.cfg")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip().lower() in ("home", "base-exec-prefix"):
                        candidate = os.path.join(v.strip(), "python.exe")
                        if os.path.exists(candidate):
                            return os.path.abspath(candidate)
        except OSError:
            pass
    return None


def _all_python_exes_to_register() -> list[str]:
    """Every python.exe / pythonw.exe involved in this run."""
    paths = []

    def _add(p):
        if not p:
            return
        ap = os.path.abspath(p)
        if os.path.exists(ap) and ap not in paths:
            paths.append(ap)

    # The interpreter as it sees itself
    _add(sys.executable)

    # Python 3.11+ exposes the real interpreter behind a venv stub here
    _add(getattr(sys, "_base_executable", None))

    # Older fallback: parse pyvenv.cfg
    _add(_venv_base_python())

    # Also pin pythonw.exe siblings of each path above
    for p in list(paths):
        pw = os.path.join(os.path.dirname(p), "pythonw.exe")
        _add(pw)

    return paths


def ensure_high_performance_gpu(auto_relaunch: bool = True) -> None:
    """Make sure every python.exe involved in this run is pinned to the dGPU.

    Call this BEFORE pygame / SpoutGL / anything that creates a GL context.
    """
    if sys.platform != "win32":
        return

    try:
        import winreg  # noqa: F401
    except ImportError:
        return

    targets = _all_python_exes_to_register()

    print("[GPU] Registering Python executables for high-performance GPU:")
    changed_paths = []
    for exe in targets:
        before = _read_registry_preference(exe)
        if _write_registry_preference(exe):
            after = _read_registry_preference(exe)
            print(f"[GPU]   SET   {exe}")
            print(f"[GPU]         before={before!r} -> after={after!r}")
            changed_paths.append(exe)
        else:
            print(f"[GPU]   OK    {exe}  (already {before!r})")

    # Already-set runs print a single OK line per exe, then return without re-execing.
    if not changed_paths:
        return

    if not auto_relaunch:
        print("[GPU] Preference set. Restart the app for it to take effect.")
        return

    # Only re-exec once per launch — prevent infinite loop if registry write
    # appears successful but the relaunched process still doesn't pick it up.
    if os.environ.get("FACE_MASK_GPU_RELAUNCHED") == "1":
        print("[GPU] Already relaunched once and the value is still being changed.")
        print("[GPU] Something is rewriting the registry. Skipping further relaunch.")
        return

    print("[GPU] Relaunching to pick up new GPU preference...")
    sys.stdout.flush()
    env = os.environ.copy()
    env["FACE_MASK_GPU_RELAUNCHED"] = "1"
    try:
        # os.execve replaces this process with a new launch (Windows-style),
        # which is what we need — the GPU preference is read at launch time.
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    except OSError as e:
        print(f"[GPU] Re-exec failed: {e}. Please restart the app manually.")
