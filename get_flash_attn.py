import sys
import platform
import re
import urllib.request
import json
import subprocess
from packaging import version

def get_system_info():
    print("Checking system environment...")
    
    # 1. Check Python Version
    py_ver = sys.version_info
    py_tag = f"cp{py_ver.major}{py_ver.minor}" # e.g., cp310
    print(f"   Python: {py_ver.major}.{py_ver.minor} ({py_tag})")

    # 2. Check PyTorch Version
    try:
        import torch
        torch_ver_raw = torch.__version__
        # Parse standard version (ignore +cu118 etc for the main version check)
        torch_ver = version.parse(torch_ver_raw.split('+')[0])
        torch_tag = f"torch{torch_ver.major}.{torch_ver.minor}" # e.g., torch2.1
        
        # 3. Check CUDA Version (from Torch)
        cuda_ver_raw = torch.version.cuda
        if not cuda_ver_raw:
            print("   [!] Error: PyTorch is not compiled with CUDA support.")
            sys.exit(1)
            
        # Clean cuda version
        # Official wheels often use 'cu12' for 12.x, or 'cu118' for 11.8
        cuda_major = cuda_ver_raw.split('.')[0]
        cuda_minor = cuda_ver_raw.split('.')[1]
        
        cuda_tag_loose = f"cu{cuda_major}"          # cu12
        cuda_tag_strict = f"cu{cuda_major}{cuda_minor}" # cu121
        
        print(f"   PyTorch: {torch_ver_raw} ({torch_tag})")
        print(f"   CUDA: {cuda_ver_raw} ({cuda_tag_loose} / {cuda_tag_strict})")
        
        return {
            "py_tag": py_tag,
            "torch_tag": torch_tag,
            "cuda_tags": [cuda_tag_strict, cuda_tag_loose], # Priority: strict > loose
            "platform": "linux_x86_64"
        }

    except ImportError:
        print("   [!] Error: PyTorch is not installed.")
        sys.exit(1)

def find_wheel_official(system_info):
    print("\n[1/2] Querying Official Dao-AILab Releases...")
    api_url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest"
    
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            print(f"   Latest Official Release: {data['tag_name']}")
            
            candidates = []
            for asset in data['assets']:
                name = asset['name']
                if not name.endswith(".whl"): continue
                if system_info['platform'] not in name: continue
                if f"-{system_info['py_tag']}-" not in name: continue
                if system_info['torch_tag'] not in name: continue
                
                # Check CUDA
                matched_cuda = False
                for c_tag in system_info['cuda_tags']:
                    if f"+{c_tag}" in name:
                        matched_cuda = True
                        break
                if not matched_cuda: continue

                # Check ABI
                if "cxx11abiFALSE" in name:
                     candidates.insert(0, asset)
                elif "cxx11abiTRUE" in name:
                     candidates.append(asset)

            if candidates:
                return candidates[0]
            else:
                print("   [-] No matching official wheel found.")
                return None

    except Exception as e:
        print(f"   [!] Failed to query official repo: {e}")
        return None

def find_wheel_community(system_info):
    print("\n[2/2] Querying Community (mjun0812) Releases...")
    # This repo often has newer builds (Torch 2.4/2.5+)
    # We check 'latest' first, but sometimes they use specific tags. 
    # For now, 'latest' is the safest dynamic check.
    api_url = "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases/latest"
    
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            print(f"   Latest Community Release: {data['tag_name']}")
            
            for asset in data['assets']:
                name = asset['name']
                # Naming convention is slightly different:
                # flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
                # (No cxx11abi tag usually)

                if not name.endswith(".whl"): continue
                if system_info['platform'] not in name: continue
                if f"-{system_info['py_tag']}-" not in name: continue
                if system_info['torch_tag'] not in name: continue

                # Check CUDA (Strict matching preferred here, e.g. cu124)
                matched_cuda = False
                for c_tag in system_info['cuda_tags']:
                    if f"+{c_tag}" in name:
                        matched_cuda = True
                        break
                if matched_cuda:
                    return asset
            
            print("   [-] No matching community wheel found.")
            return None

    except Exception as e:
        print(f"   [!] Failed to query community repo: {e}")
        return None

def main():
    info = get_system_info()
    
    # Try official first
    wheel = find_wheel_official(info)
    
    # Try community fallback if official fails
    if not wheel:
        wheel = find_wheel_community(info)
    
    if wheel:
        print(f"\n[+] Found compatible wheel: {wheel['name']}")
        print(f"    Source: {wheel.get('url', 'Unknown').split('/repos/')[1].split('/')[0] if 'url' in wheel else 'Community'}")
        print(f"    Download URL: {wheel['browser_download_url']}")
        
        choice = input("\nDo you want to install this with pip now? (y/n): ")
        if choice.lower() == 'y':
            print("Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", wheel['browser_download_url']])
            print("\n[+] Installation complete!")
        else:
            print("\nSkipping installation.")
    else:
        print("\n[-] No compatible wheel found in either repository.")
        print("    You may need to build from source using the safe flags:")
        print("    export MAX_JOBS=1 && pip install . --no-build-isolation")

if __name__ == "__main__":
    main()