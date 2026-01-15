# piviz/core/gpu_selector.py
"""
PiViz GPU Auto-Selection
========================

Automatically selects the best available GPU for rendering.
Priority: NVIDIA > AMD > Intel > Software
"""

import os
import sys
import subprocess
from typing import Optional, Tuple, List

# --- VISUAL STYLING CONSTANTS ---
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_GREY = "\033[90m"
C_RED = "\033[91m"
C_RESET = "\033[0m"


def get_available_gpus() -> List[dict]:
    """Detect available GPUs on the system."""
    gpus = []

    # Method 1: Try lspci (Linux)
    try:
        result = subprocess.run(
            ['lspci', '-nn'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if 'vga' in line_lower or '3d' in line_lower or 'display' in line_lower:
                    # Cleanup name for display
                    raw_name = line.split(':')[-1].strip()
                    # Remove common bracket noise
                    raw_name = raw_name.split('[')[0].strip()

                    gpu_info = {'name': raw_name, 'vendor': 'unknown', 'type': 'unknown', 'priority': 0}

                    if 'nvidia' in line_lower:
                        gpu_info['vendor'] = 'nvidia';
                        gpu_info['type'] = 'discrete';
                        gpu_info['priority'] = 100
                    elif 'amd' in line_lower or 'radeon' in line_lower:
                        gpu_info['vendor'] = 'amd';
                        gpu_info['type'] = 'discrete';
                        gpu_info['priority'] = 90
                    elif 'intel' in line_lower:
                        gpu_info['vendor'] = 'intel';
                        gpu_info['type'] = 'integrated';
                        gpu_info['priority'] = 50

                    gpus.append(gpu_info)
    except:
        pass

    # Method 2: Try nvidia-smi
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
    except:
        pass

    return gpus


def check_prime_available() -> bool:
    try:
        return subprocess.run(['prime-select', 'query'], capture_output=True).returncode == 0
    except:
        return False


def get_current_prime_profile() -> Optional[str]:
    try:
        res = subprocess.run(['prime-select', 'query'], capture_output=True, text=True)
        if res.returncode == 0: return res.stdout.strip()
    except:
        pass
    return None


def set_nvidia_offload_env():
    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['__VK_LAYER_NV_optimus'] = 'NVIDIA_only'


def set_amd_offload_env():
    os.environ['DRI_PRIME'] = '1'


def auto_select_gpu(verbose: bool = True) -> dict:
    """
    Automatically select the best available GPU.
    """
    result = {
        'selected_gpu': None, 'method': None,
        'env_vars_set': [], 'available_gpus': []
    }

    gpus = get_available_gpus()
    result['available_gpus'] = gpus

    # Print Header
    if verbose:
        print(f"\n{C_BLUE}╔══════════════════════════════════════════════════════╗")
        print(f"║ {C_GREEN}GPU Auto-Select{C_BLUE}                                  ║")
        print(f"╚══════════════════════════════════════════════════════╝{C_RESET}")

    if not gpus:
        if verbose:
            print(f" {C_GREY}► Status:{C_RESET}   {C_RED}No GPUs detected{C_RESET}")
        return result

    # Sort
    gpus.sort(key=lambda g: g['priority'], reverse=True)

    if verbose:
        for g in gpus:
            name_clean = g['name'][:45] + "..." if len(g['name']) > 45 else g['name']
            print(f" {C_GREY}► Detect:{C_RESET}   {name_clean}")

    # Logic
    best_gpu = gpus[0]
    result['selected_gpu'] = best_gpu

    has_nvidia = any(g['vendor'] == 'nvidia' for g in gpus)
    has_amd = any(g['vendor'] == 'amd' and g['type'] == 'discrete' for g in gpus)
    has_intel = any(g['vendor'] == 'intel' for g in gpus)

    status_msg = ""

    if has_nvidia and (has_intel or len(gpus) > 1):
        if get_current_prime_profile() == 'nvidia':
            result['method'] = 'prime-select'
            status_msg = "NVIDIA (Prime Active)"
        else:
            set_nvidia_offload_env()
            result['env_vars_set'] = ['__NV_PRIME_RENDER_OFFLOAD']
            result['method'] = 'offload'
            status_msg = "NVIDIA (Env Offload)"

    elif has_amd and has_intel:
        set_amd_offload_env()
        result['env_vars_set'] = ['DRI_PRIME']
        result['method'] = 'offload'
        status_msg = "AMD (DRI_PRIME)"

    elif has_nvidia:
        result['method'] = 'default'
        status_msg = "NVIDIA (Default)"

    else:
        result['method'] = 'default'
        status_msg = f"{best_gpu['vendor'].capitalize()} (Default)"

    if verbose:
        print(f" {C_GREY}► Select:{C_RESET}   {C_GREEN}{status_msg}{C_RESET}")
        print(f"\n")

    return result


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
    """Print detailed GPU info in the main style."""
    print(f"\n{C_BLUE}╔══════════════════════════════════════════════════════╗")
    print(f"║ {C_GREEN}System Diagnostics{C_BLUE}                                 ║")
    print(f"╚══════════════════════════════════════════════════════╝{C_RESET}")

    # 1. Hardware
    gpus = get_available_gpus()
    if not gpus:
        print(f" {C_GREY}► Hardw:{C_RESET}    {C_RED}No GPUs Found{C_RESET}")
    else:
        for i, g in enumerate(gpus):
            tag = "Hardw:" if i == 0 else "      "
            print(f" {C_GREY}► {tag}{C_RESET}    {g['name']}")

    # 2. Config
    prime = get_current_prime_profile()
    if prime:
        print(f" {C_GREY}► Prime:{C_RESET}    {prime}")

    # 3. Context
    info = verify_gpu_selection()
    print(f" {C_GREY}► Render:{C_RESET}   {info['renderer']}")
    print(f" {C_GREY}► Vendor:{C_RESET}   {info['vendor']}")
    print(f" {C_GREY}► OpenGL:{C_RESET}   {info['version']}")
    print(f"\n")


# ============================================================
# AUTO-INIT
# ============================================================

_gpu_selection_done = False


def ensure_gpu_selected():
    global _gpu_selection_done
    if not _gpu_selection_done:
        auto_select_gpu(verbose=False)
        _gpu_selection_done = True


if __name__ == '__main__':
    # Run diagnostics
    auto_select_gpu(verbose=True)
    print_gpu_info()
