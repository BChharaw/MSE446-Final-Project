FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN tar -xzf train-clean-100.tar.gz -C /data
    # Set working directory
WORKDIR /workspace

# Install necessary tools
RUN apt-get update && \
    apt-get install -y libsndfile1 git sudo ffmpeg wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user that matches host UID/GID (will be set at runtime)
RUN useradd -m -s /bin/bash vscodeuser && \
    echo "vscodeuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscodeuser

# Set default command
CMD ["/bin/bash"]
