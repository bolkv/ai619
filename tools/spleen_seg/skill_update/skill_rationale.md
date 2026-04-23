# Skill Rationale

## Why this tool should exist
- 현재 ChatClinic은 VCF, DICOM, NIfTI, 이미지 등 다양한 의료 데이터를 입력으로 받지만, 업로드된 의료 영상에서 관심 구조물을 분할하는 segmentation 기능은 제공되지 않는다. `nifti_review_tool`이 파일 메타데이터와 미리보기를 제공하더라도, 그다음 단계인 병변/장기 추출은 사용자가 외부 도구를 따로 돌려야만 가능하다. 임상 연구와 교육 환경에서 segmentation은 volumetric 측정, 병변 정량화, 치료 반응 평가 등 거의 모든 후속 분석의 전제가 되기 때문에, segmentation 관련 tool이 반드시 필요하다.
- `spleen_seg` (MONAI UNet, spleen_ct bundle): 3D 복부 CT에서 비장 2-class 마스크를 생성하여 비장 볼륨 측정 등 단일 장기 분석에 활용한다.
- 본 패치에서 활용되는 MONAI는 NVIDIA가 주도하는 PyTorch 기반 의료 영상 딥러닝 오픈소스 프레임워크로, 사전학습된 segmentation 모델을 재현 가능한 bundle 형태로 배포한다.

## Why the orchestrator should call it
- 사용자가 의료 영상을 업로드한 뒤 segmentation을 명시적으로 요청할 때(`@seg_spleen`), orchestrator는 입력된 키워드를 tool.json의 alias와 매칭하여 해당 tool을 호출한다. 이때 tool.json의 `source_types` 필드에 따라 현재 업로드된 소스 타입과 맞지 않으면 시스템이 자동으로 호출을 차단하므로, 잘못된 modality에 tool이 실행되는 상황을 막을 수 있다. 영상 분할은 단순 메타데이터 응답이나 LLM의 서술로는 대체 불가능한 연산 과정이므로, orchestrator가 직접 중개해 tool을 실행시키는 것이 유일하게 재현 가능하고 정량적인 결과를 얻는 방식이다.
- 본 tool은 고유한 alias(`seg_spleen`)를 가지도록 `trigger_keywords`에 등록되어 있어, 같은 키워드에 여러 tool이 매칭되는 충돌 없이 1:1로 라우팅된다. 사용자는 업로드한 영상의 modality/타겟에 맞는 alias를 고르기만 하면 되고, source_type 검증은 tool.json 선언을 통해 시스템이 대신 처리한다.
  - 복부 CT: `@seg_spleen` (`source_types: ["nifti"]`)
- 각 tool.json의 `source_types`, `routing.trigger_keywords` 필드가 orchestrator의 라우팅 근거로 사용되므로, orchestrator는 파일을 열지 않고도 manifest 정보만으로 적절한 tool을 선택할 수 있다.
- 본 tool은 결정론적인 inference 파이프라인을 거쳐 실제 마스크 파일을 artifact로 반환하므로, orchestrator는 영상 분할 요청을 이 tool로 처리함으로써 임상/연구 맥락에서 신뢰할 수 있는 형태의 결과를 확보할 수 있다.

## Why approval is or is not required
- approval required. 다음 세 가지 이유 때문에 사용자 승인 후 실행한다.
  - GPU 기반 3D 딥러닝 inference를 수행하기 때문에 자원 소모가 크다.
  - segmentation 마스크는 이후 볼륨 측정이나 병변 정량화처럼 임상적 판단으로 이어질 수 있는 결과물이다. 그렇기 때문에 사용자가 실행 대상과 파라미터를 한 번 더 확인한 뒤 수행하는 것이 안전하다.
  - NIfTI 파일 메타데이터만으로는 MRI인지 CT인지 확정할 수 없다. 잘못된 조합으로 실행될 경우 무의미한 결과가 나올 수 있으므로, 승인 단계에서 입력과 도구가 맞는지 다시 점검한다.

## What educational value it adds
- 동일한 ChatClinic 인터페이스 안에서 3D CT modality와 UNet 아키텍처가 어떤 타겟에 어떻게 사용되는지를 확인할 수 있다. `@seg_spleen`을 통해 2-class 비장 출력 마스크 구조를 확인할 수 있다.
- `tool.json` + `logic.py` + skill patch로 구성되는 ChatClinic plugin 패턴을 실습할 수 있어, 향후 원하는 tool을 추가할 때 그대로 템플릿으로 사용 가능하다.
- segmentation 결과가 `segmentation_result` slot을 통해 Studio로 렌더링되는 흐름을 경험함으로써, 업로드 → tool 호출 → 결과 해석으로 이어지는 end-to-end 의료 AI 워크플로우를 이해할 수 있다.