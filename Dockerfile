FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git \
      build-essential g++ gcc ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      libhts-dev zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt .
# Install CUDA 12.1 PyTorch wheels first so downstream packages (e.g. vllm) pick them up.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install \
      "fastapi>=0.115,<1.0" \
      "uvicorn>=0.30,<1.0" \
      "python-multipart>=0.0.9" \
      "pydantic>=2.8,<3.0" \
      "pysam>=0.23,<1.0" \
      "openpyxl>=3.1,<4.0" \
      "Pillow>=10.0,<12.0" \
      "nibabel>=5.2,<6.0" \
      "transformers==4.57.6" \
      "vllm==0.18.1"

RUN nvcc --version && \
    python -c "import torch; print(torch.__version__, torch.version.cuda)"

COPY . /workspace

CMD ["bash"]
