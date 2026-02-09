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
            print("   [!] Error: PyTorch is not compiled with CUDA support. Cannot install FlashAttention.")
            sys.exit(1)
            
        # Clean cuda version to match wheel naming (e.g. 12.1 -> cu12)
        # Note: FlashAttn wheels often group 12.x as 'cu12' and 11.x as 'cu11'
        cuda_major = cuda_ver_raw.split('.')[0]
        cuda_tag_loose = f"cu{cuda_major}" # cu12
        cuda_tag_strict = f"cu{cuda_major}{cuda_ver_raw.split('.')[1]}" # cu121
        
        print(f"   PyTorch: {torch_ver_raw} ({torch_tag})")
        print(f"   CUDA: {cuda_ver_raw} ({cuda_tag_loose} / {cuda_tag_strict})")
        
        return {
            "py_tag": py_tag,
            "torch_tag": torch_tag,
            "cuda_tags": [cuda_tag_loose, cuda_tag_strict], # Try strict first, then loose
            "platform": "linux_x86_64" # FlashAttn wheels are mostly Linux only
        }

    except ImportError:
        print("   [!] Error: PyTorch is not installed. Please install PyTorch first.")
        sys.exit(1)

def find_wheel(system_info):
    print("\nQuerying GitHub for latest FlashAttention release...")
    api_url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases/latest"
    
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            release_tag = data['tag_name']
            assets = data['assets']
            print(f"   Latest Release: {release_tag}")
            
            # Filter assets
            candidates = []
            for asset in assets:
                name = asset['name']
                if not name.endswith(".whl"):
                    continue
                
                # Check Platform
                if system_info['platform'] not in name:
                    continue
                
                # Check Python
                if f"-{system_info['py_tag']}-" not in name:
                    continue
                
                # Check PyTorch
                if system_info['torch_tag'] not in name:
                    continue
                
                # Check CUDA (Try to match either cu121 or cu12)
                # Recent wheels often use 'cu12' for all 12.x versions
                matched_cuda = False
                for c_tag in system_info['cuda_tags']:
                    if f"+{c_tag}" in name:
                        matched_cuda = True
                        break
                if not matched_cuda:
                    continue

                # Check ABI (We usually want FALSE for standard PyTorch)
                # If users built from source with CXX11_ABI=1, they need TRUE, but FALSE is safe default
                if "cxx11abiFALSE" in name:
                     candidates.insert(0, asset) # Prioritize FALSE
                elif "cxx11abiTRUE" in name:
                     candidates.append(asset)

            if not candidates:
                return None
            
            return candidates[0] # Return best match

    except Exception as e:
        print(f"   [!] Failed to query GitHub: {e}")
        return None

def main():
    info = get_system_info()
    wheel = find_wheel(info)
    
    if wheel:
        print(f"\n[+] Found compatible wheel: {wheel['name']}")
        print(f"    Download URL: {wheel['browser_download_url']}")
        
        choice = input("\nDo you want to install this with pip now? (y/n): ")
        if choice.lower() == 'y':
            print("Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", wheel['browser_download_url']])
            print("\n[+] Installation complete!")
        else:
            print("\nSkipping installation.")
    else:
        print("\n[-] No exact pre-built wheel found for this specific environment.")
        print("    You might need to build from source (use MAX_JOBS=1) or check the release page manually.")
        print("    https://github.com/Dao-AILab/flash-attention/releases")

if __name__ == "__main__":
    main()