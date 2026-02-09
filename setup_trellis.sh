# for 5090 series, 12.0 compute capability
export TORCH_CUDA_ARCH_LIST="12.0"
export NVCC_THREADS=2

sudo apt-get install -y libeigen3-dev
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
pip install ninja
pip install psutil
pip install fastapi uvicorn
cd TRELLIS.2
MAX_JOBS=12 pip install flash-attn --no-build-isolation --extra-index-url https://pypi.nvidia.com
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm