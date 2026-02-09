# for 5090 series, 12.0 compute capability
export TORCH_CUDA_ARCH_LIST="12.0"
export NVCC_THREADS=2

sudo apt-get install -y libeigen3-dev
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
pip install ninja
pip install psutil xformers
pip install fastapi uvicorn
python3 get_flash_attn.py
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm