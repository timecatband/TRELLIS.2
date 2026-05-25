# Respect an existing override, otherwise compile local CUDA extensions for the
# visible GPU instead of assuming RTX 50-series / Blackwell.
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    DETECTED_CUDA_ARCH=$(python3 -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "")' 2>/dev/null)
    if [ -n "$DETECTED_CUDA_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$DETECTED_CUDA_ARCH"
    fi
fi
export NVCC_THREADS=2

sudo apt-get install -y libeigen3-dev
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
pip install ninja
pip install psutil xformers
pip install fastapi uvicorn
python3 get_flash_attn.py --version "${FLASH_ATTN_VERSION:-2.7.3}" || exit 1
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
