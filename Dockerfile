# BBDOS Docker Image
# Reproducible environment for sparse 2-bit neural computation
#
# Quick start:
#   docker build -t bbdos .
#   docker run bbdos
#
# With GPU:
#   docker run --gpus all bbdos

# Use multi-platform base
FROM python:3.11-slim

# Metadata
LABEL maintainer="Aaron (Tripp) Josserand-Austin <iam@anjaustin.com>"
LABEL version="1.0.0"
LABEL description="BBDOS: 2-Bit Conditional Ternary Neural Architecture"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

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

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default: run demo to verify core claim
CMD ["python", "scripts/demo.py"]

# Other commands:
#   docker run bbdos python -m pytest tests/ -v
#   docker run bbdos python scripts/benchmark.py
#   docker run -it bbdos bash
