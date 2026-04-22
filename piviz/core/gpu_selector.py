"""
PiViz GPU Auto-Selection
========================

Automatically selects the best available GPU for rendering.
Priority: NVIDIA > AMD > Intel > Software
"""

import os
import re
import sys
import subprocess
from typing import Optional, List

# ── Terminal style (matches PiViz banner) ────────────────────────────────────
_DIM = "\033[90m"
_WHITE = "\033[97m"
_ORANGE = "\033[38;5;208m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"

_W = 74
_ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _vlen(s: str) -> int:
    return len(_ansi.sub('', s))


def _row(content: str) -> str:
    pad = _W - 2 - _vlen(content)
    return f"{_DIM}│{_RESET} {content}{' ' * max(0, pad - 1)}{_DIM}│{_RESET}"


def _div() -> str:
    return f"{_DIM}├{'─' * (_W - 2)}┤{_RESET}"


def _top() -> str:
    return f"{_DIM}╭{'─' * (_W - 2)}╮{_RESET}"


def _bot() -> str:
    return f"{_DIM}╰{'─' * (_W - 2)}╯{_RESET}"


# ── GPU detection ─────────────────────────────────────────────────────────────

def get_available_gpus() -> List[dict]:
    """Detect available GPUs on the system."""
    gpus = []

    # Method 1: lspci (Linux)
    try:
        result = subprocess.run(
            ['lspci', '-nn'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                ll = line.lower()
                if not ('vga' in ll or '3d' in ll or 'display' in ll):
                    continue

                # Parse: "... controller [class]: Vendor Description [chip] [xxxx:xxxx] (rev N)"
                # Split on ']: ' to get everything after the class code
                parts = line.split(']: ', 1)
                if len(parts) > 1:
                    raw = parts[1]
                    # Strip trailing [vendor:device] (rev N)
                    raw = re.sub(r'\s*\[[\da-f]{4}:[\da-f]{4}\].*$', '', raw, flags=re.IGNORECASE).strip()
                else:
                    raw = line.split(':')[-1].strip()

                gpu = {'name': raw, 'vendor': 'unknown', 'type': 'unknown', 'priority': 0}
                if 'nvidia' in ll:
                    gpu.update(vendor='nvidia', type='discrete', priority=100)
                elif 'amd' in ll or 'radeon' in ll:
                    gpu.update(vendor='amd', type='discrete', priority=90)
                elif 'intel' in ll:
                    gpu.update(vendor='intel', type='integrated', priority=50)

                gpus.append(gpu)
    except Exception:
        pass

    # Method 2: nvidia-smi fallback
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for name in result.stdout.strip().split('\n'):
                if name and not any(g['vendor'] == 'nvidia' for g in gpus):
                    gpus.append({
                        'name': f'NVIDIA {name}', 'vendor': 'nvidia',
                        'type': 'discrete', 'priority': 100
                    })
    except Exception:
        pass

    return gpus


def check_prime_available() -> bool:
    try:
        return subprocess.run(['prime-select', 'query'], capture_output=True).returncode == 0
    except Exception:
        return False


def get_current_prime_profile() -> Optional[str]:
    try:
        res = subprocess.run(['prime-select', 'query'], capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def set_nvidia_offload_env():
    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['__VK_LAYER_NV_optimus'] = 'NVIDIA_only'


def set_amd_offload_env():
    os.environ['DRI_PRIME'] = '1'


def probe_default_renderer() -> dict:
    """Spawn a clean subprocess to detect the active renderer without env overrides."""
    code = (
        "import moderngl; "
        "ctx=moderngl.create_context(standalone=True); "
        "print(ctx.info['GL_VENDOR']); "
        "print(ctx.info['GL_RENDERER'])"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=5,
            env=os.environ.copy()
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                return {'vendor': lines[0].strip(), 'renderer': lines[1].strip()}
    except Exception:
        pass
    return {'vendor': 'Unknown', 'renderer': 'Unknown'}


# ── Main selection logic ──────────────────────────────────────────────────────

def auto_select_gpu(verbose: bool = True) -> dict:
    """
    1. Probe the active renderer
    2. If already on the best GPU → do nothing
    3. If discrete GPU exists but inactive → force offload env vars
    """
    result = {
        'selected_gpu': None,
        'method': 'unknown',
        'env_vars_set': [],
        'available_gpus': []
    }

    gpus = get_available_gpus()
    result['available_gpus'] = gpus

    if not gpus:
        if verbose:
            print()
            print(_top())
            print(_row(f"{_ORANGE}GPU{_RESET}  {_DIM}·  no GPU detected{_RESET}"))
            print(_bot())
            print()
        return result

    has_nvidia_hw = any(g['vendor'] == 'nvidia' for g in gpus)
    has_amd_hw = any(g['vendor'] == 'amd' and g['type'] == 'discrete' for g in gpus)

    default_renderer = probe_default_renderer()
    default_vendor = default_renderer['vendor'].lower()

    is_already_nvidia = 'nvidia' in default_vendor
    is_already_amd = 'amd' in default_vendor or 'ati' in default_vendor

    # Decision
    if is_already_nvidia:
        result.update(method='native_default',
                      selected_gpu=next((g for g in gpus if g['vendor'] == 'nvidia'), None))
        status = f"{_WHITE}NVIDIA{_RESET}  {_DIM}native{_RESET}"

    elif has_nvidia_hw:
        set_nvidia_offload_env()
        result.update(method='forced_offload',
                      env_vars_set=['__NV_PRIME_RENDER_OFFLOAD', '__GLX_VENDOR_LIBRARY_NAME'],
                      selected_gpu=next((g for g in gpus if g['vendor'] == 'nvidia'), None))
        status = f"{_GREEN}NVIDIA{_RESET}  {_DIM}offload enabled{_RESET}"

    elif is_already_amd:
        result.update(method='native_default',
                      selected_gpu=next((g for g in gpus if g['vendor'] == 'amd'), None))
        status = f"{_WHITE}AMD{_RESET}  {_DIM}native{_RESET}"

    elif has_amd_hw:
        set_amd_offload_env()
        result.update(method='forced_offload',
                      env_vars_set=['DRI_PRIME'],
                      selected_gpu=next((g for g in gpus if g['vendor'] == 'amd'), None))
        status = f"{_GREEN}AMD{_RESET}  {_DIM}DRI_PRIME enabled{_RESET}"

    else:
        result['method'] = 'system_default'
        status = f"{_DIM}{default_renderer['vendor']}  ·  system default{_RESET}"

    if verbose:
        print()
        print(_top())
        print(_row(f"{_ORANGE}GPU{_RESET}  {_DIM}·  Smart Detection{_RESET}"))
        print(_div())
        for g in gpus:
            print(_row(f"{_DIM}{g['name']}{_RESET}"))
        print(_div())
        renderer_line = default_renderer['renderer']
        print(_row(f"{_DIM}{renderer_line}{_RESET}"))
        print(_row(status))
        print(_bot())
        print()

    return result


# ── Diagnostics ───────────────────────────────────────────────────────────────

def verify_gpu_selection() -> dict:
    result = {'renderer': 'Unknown', 'vendor': 'Unknown', 'version': 'Unknown'}
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        result['renderer'] = ctx.info.get('GL_RENDERER', 'Unknown')
        result['vendor'] = ctx.info.get('GL_VENDOR', 'Unknown')
        result['version'] = ctx.info.get('GL_VERSION', 'Unknown')
        ctx.release()
    except Exception as e:
        result['error'] = str(e)
    return result


def print_gpu_info():
    """Print detailed GPU diagnostics in PiViz banner style."""
    gpus = get_available_gpus()
    info = verify_gpu_selection()
    prime = get_current_prime_profile()

    print()
    print(_top())
    print(_row(f"{_ORANGE}GPU Diagnostics{_RESET}"))
    print(_div())
    if gpus:
        for g in gpus:
            print(_row(f"{_DIM}{g['name']}{_RESET}"))
    else:
        print(_row(f"{_RED}no GPUs found{_RESET}"))
    if prime:
        print(_row(f"{_DIM}Prime  ·  {_RESET}{prime}"))
    print(_div())
    print(_row(f"{_DIM}Renderer  ·  {_RESET}{info['renderer']}"))
    print(_row(f"{_DIM}Vendor    ·  {_RESET}{info['vendor']}"))
    print(_row(f"{_DIM}OpenGL    ·  {_RESET}{info['version']}"))
    print(_bot())
    print()


# ── Auto-init guard ───────────────────────────────────────────────────────────

_gpu_selection_done = False


def ensure_gpu_selected():
    global _gpu_selection_done
    if not _gpu_selection_done:
        auto_select_gpu(verbose=False)
        _gpu_selection_done = True


if __name__ == '__main__':
    auto_select_gpu(verbose=True)
    print_gpu_info()
