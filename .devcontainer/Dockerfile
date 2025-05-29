# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy all code (including pyproject.toml, setup.py, README.md, and the czbi_da_parez package)
COPY . .

# Install pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Default command
CMD ["/bin/bash"]
