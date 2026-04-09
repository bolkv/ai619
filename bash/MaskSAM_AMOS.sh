#!/usr/bin/env bash
# Training recipe for Dataset052 (AMOS CT, MaskSAM trainer).
# Uses the unified datasets/ layout. nnUNet source is vendored under
# tools/nnunet_amos/vendor/ and must be importable (pip install -e it, or
# add its path to PYTHONPATH).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export nnUNet_raw="${REPO_ROOT}/datasets/nnunet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/datasets/nnunet_preprocessed"
export nnUNet_results="${REPO_ROOT}/datasets/nnunet_results"
export PYTHONPATH="${REPO_ROOT}/tools/nnunet_amos/vendor:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0   nnUNetv2_train 52 3d_fullres 0 -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 1 -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 2 --c
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 2 -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 2 --c
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 3 -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 2 --c
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 52 3d_fullres 4 -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 2 --c
