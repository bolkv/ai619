# Skill Patch Proposal

## Tool
- `lung_seg`

## Purpose
- `lung_seg`: Chest X-ray 폐 segmentation. torchxrayvision PSPNet을 사용하여 2D CXR 이미지에서 폐 영역 마스크를 생성한다.

## When to use
- `lung_seg` (`@seg_lung`): 사용자가 PNG/JPG 형식의 Chest X-ray 이미지를 업로드한 상태에서 폐 영역 segmentation을 명시적으로 요청할 때 사용한다.

## When not to use
- 사용자가 segmentation이 아닌 일반적인 영상 메타데이터 확인, 시각화만 요청하는 경우에는 사용하지 않는다. 이 경우 `image_review_tool`을 사용한다.
- 모달리티가 맞지 않는 경우 사용하지 않는다.
- 사용자가 segmentation에 대해 개념적인 질문만 하는 경우에는 tool을 실행하지 않고 일반 지식으로 답변한다.
- 소스 파일이 업로드되지 않은 상태에서는 사용하지 않는다.

## Source type
- image

## Modality
- 2D X-ray (PNG/JPG)

## Recommended stage
- on-demand

## Depends on
- `image_review_tool` (업로드 시 자동 실행되어 이미지 메타데이터 제공)

## Runtime
- host compatible: gpu
- supported accelerators: gpu
- preferred accelerator: gpu
- requires gpu: yes
- cpu fallback allowed: no

## Approval policy
- approval required

## Produces
- `segmentation_result` (폐 영역 마스크 PNG)

## Example routing note
- PNG/JPG 업로드 시 먼저 `image_review_tool`이 자동 실행되어 이미지 메타데이터를 채운 후, CXR 임상 맥락에서 `@seg_lung`이 호출될 때만 해당 tool로 라우팅한다.
- 사용자 의도(입력한 alias)를 기준으로 라우팅하며 `source_types` 필드로 최소한의 파일 형식 안전장치를 건다.