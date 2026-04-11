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
      libglib2.0-0 libsm6 libxext6 libxrender1 graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.2 torchvision==0.16.2 \
 && pip install -r requirements.txt

COPY tools/multi_organ_seg/vendor /workspace/tools/multi_organ_seg/vendor

RUN nvcc --version && \
    python -c "import torch; print(torch.__version__, torch.version.cuda)" && \
    pip install -e /workspace/tools/multi_organ_seg/vendor --no-build-isolation

COPY . /workspace

ENV nnUNet_raw=/workspace/tools/multi_organ_seg/dataset/nnunet_raw \
    nnUNet_preprocessed=/workspace/tools/multi_organ_seg/dataset/nnunet_preprocessed \
    nnUNet_results=/workspace/tools/multi_organ_seg/weights/nnunet \
    SAM_CHECKPOINT=/workspace/tools/multi_organ_seg/weights/sam/sam_vit_h_4b8939.pth


CMD ["bash"]