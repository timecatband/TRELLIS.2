sudo apt-get install -y libeigen3-dev
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
pip install ninja
cd TRELLIS.2
bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel \
--flexgemm