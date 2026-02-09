sudo apt-get install -y libeigen3-dev
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
pip install ninja
pip install psutil
pip install fastapi uvicorn
cd TRELLIS.2
pip install flash-attn --no-build-isolation --extra-index-url https://pypi.nvidia.com
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm