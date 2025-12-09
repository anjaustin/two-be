# BBDOS Docker Image
# Reproducible environment for sparse 2-bit neural computation
#
# Build: docker build -t bbdos:latest .
# Run:   docker run --gpus all -it bbdos:latest

# Base image: NVIDIA L4T with PyTorch for Jetson
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Metadata
LABEL maintainer="Tripp & Double-D"
LABEL version="1.0.0"
LABEL description="BBDOS: BitSwitch-Based Distributed Operating System"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build the BitSwitch kernel
RUN cd bbdos/kernel && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Alternative: Use legacy kernel build if present
RUN if [ -f /workspace/BBDOS/build/libbitswitch.so ]; then \
        cp /workspace/BBDOS/build/libbitswitch.so bbdos/kernel/build/; \
    fi

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Useful commands:
# - Run tests: docker run --gpus all bbdos:latest
# - Train CPU: docker run --gpus all bbdos:latest python scripts/train_cpu.py
# - Benchmark: docker run --gpus all bbdos:latest python scripts/benchmark.py
# - Interactive: docker run --gpus all -it bbdos:latest bash
