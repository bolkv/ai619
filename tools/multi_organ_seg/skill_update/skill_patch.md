# Skill Patch Proposal

## Tool
- `multi_organ_seg`

## Purpose
- `multi_organ_seg`: AMOS CT 다장기 segmentation. MaskSAM 기반 nnU-Net v2를 사용하여 3D CT 볼륨에서 배경 포함 16-class 마스크(15개 장기)를 생성한다.

## When to use
- `multi_organ_seg` (`@seg_organ`): 사용자가 NIfTI 형식의 복부 CT 볼륨을 업로드한 상태에서 다장기 segmentation을 명시적으로 요청할 때 사용한다. 15개 장기(간, 신장, 비장, 췌장 등)를 배경과 함께 16-class로 동시에 분할한다.

## When not to use
- 사용자가 segmentation이 아닌 일반적인 영상 메타데이터 확인, 시각화만 요청하는 경우에는 사용하지 않는다. 이 경우 `nifti_review_tool`을 사용한다.
- 모달리티가 맞지 않는 경우 사용하지 않는다. 예를 들어 MRI 볼륨에 `@seg_organ`(CT 전용)을 실행하면 잘못된 결과가 나온다.
- 사용자가 segmentation에 대해 개념적인 질문만 하는 경우에는 tool을 실행하지 않고 일반 지식으로 답변한다.
- 소스 파일이 업로드되지 않은 상태에서는 사용하지 않는다.

## Source type
- nifti

## Modality
- 3D CT (NIfTI)

## Recommended stage
- on-demand

## Depends on
- `nifti_review_tool` (업로드 시 자동 실행되어 NIfTI 메타데이터 제공)

## Runtime
- host compatible: gpu
- supported accelerators: gpu
- preferred accelerator: gpu
- requires gpu: yes
- cpu fallback allowed: no

## Approval policy
- approval required

## Produces
- `segmentation_result` (16-class 마스크 NIfTI, 배경 + 15개 장기)

## Example routing note
- NIfTI 업로드 시 먼저 `nifti_review_tool`이 자동 실행되어 파일 경로/shape/voxel 정보를 채운 후, 사용자가 `@seg_organ`(복부 CT)을 명시적으로 호출할 때만 해당 tool로 라우팅한다.
- NIfTI 메타데이터만으로는 MRI/CT 구분이 불가능하므로, 사용자 의도(입력한 alias)를 기준으로 라우팅하며 `source_types` 필드로 최소한의 파일 형식 안전장치를 건다.