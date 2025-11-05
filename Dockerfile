FROM verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ENV PATH=/miniconda/bin:${PATH}
ARG HOME=/root

# Install essential packages and dependencies
RUN apt-get update && apt-get install -y \
    locales \
    python3-pip python3-dev \
    golang-1.18 \
    git wget curl \
    zsh tmux vim htop \
    clang-format clang-tidy \
    swig \
    iputils-ping \
    netcat-openbsd \
    iproute2 lsof \
    screen \
    qtdeclarative5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install rdkit==2025.3.6
