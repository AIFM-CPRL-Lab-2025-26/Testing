FROM fedora:43

# Make sure all packages are up to date
RUN dnf update -y
# Install basic packages
RUN dnf install -y git gcc python3 python3-devel SDL2 SDL2-devel meson

# Install needed baseline Drivers
RUN dnf install -y kernel-devel-matched kernel-headers

# Install Driver Repository
RUN dnf config-manager addrepo -y --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo -y  && dnf clean expire-cache

# Install Drivers
RUN dnf install -y cuda-drivers

# Install compilers required for Numpy & pyGame
RUN dnf install -y g++ clang

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env

# Install libraries to make individual Tests run more efficiently
RUN $HOME/.local/bin/uv pip install --system nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.3.83 nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.5 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvshmem-cu12==3.3.20 nvidia-nvtx-cu12==12.8.90 torch

WORKDIR /home
