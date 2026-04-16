FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git \
      build-essential g++ gcc ninja-build \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      libhts-dev zlib1g-dev libbz2-dev liblzma-dev libcurl4-gnutls-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

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
      "transformers==4.57.6"

# ---------------------------------------------------------------------------
# Per-tool dependencies. Each tool under tools/<task>_seg/ has its own
# requirements.txt; copy them up-front (ahead of the full source copy) so
# this layer can be cached across source-only edits.
# ---------------------------------------------------------------------------
COPY tools/brain_tumor_seg/requirements.txt     /tmp/tool_reqs/brain_tumor_seg.txt
COPY tools/spleen_seg/requirements.txt          /tmp/tool_reqs/spleen_seg.txt
COPY tools/pancreas_tumor_seg/requirements.txt  /tmp/tool_reqs/pancreas_tumor_seg.txt
COPY tools/lung_seg/requirements.txt            /tmp/tool_reqs/lung_seg.txt
COPY tools/multi_organ_seg/requirements.txt     /tmp/tool_reqs/multi_organ_seg.txt
RUN set -e; \
    for f in /tmp/tool_reqs/*.txt; do \
      echo "== installing $f =="; \
      if [ -s "$f" ]; then pip install -r "$f"; fi; \
    done && \
    rm -rf /tmp/tool_reqs

RUN nvcc --version && \
    python -c "import torch; print(torch.__version__, torch.version.cuda)"

COPY . /workspace

CMD ["bash"]
