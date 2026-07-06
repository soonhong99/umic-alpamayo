# Current Scope

이 repo는 UMIC 확정분을 새 연구생이 바로 실행하고 구조를 이해할 수 있게 만든 최소 공개
runtime repo다.

## 포함하는 것

- Jetson AGX Thor에서 Alpamayo 1.5 eager vs UMIC A/B benchmark를 실행하는 스크립트
- `umic.apply(model)` 원콜 API
- 채택된 UMIC 최적화 11종
  - MLP gate/SiLU/up fusion
  - q/o projection Triton dispatch
  - RMSNorm fusion
  - residual add + norm fusion (LM)
  - text/vision RoPE fusion
  - ViT LayerNorm fusion
  - InplaceKVCache
  - per-KV-length decode CUDA Graph
  - ViT 패치 임베딩 Conv→Linear fusion (2026-07-06 추가)
  - ViT packed varlen attention fusion (2026-07-06 추가)
  - ViT residual add + LayerNorm 파이프라인 fusion (2026-07-06 추가)
- 공식 latency benchmark와 출력 등가성 문서
- Thor 환경 sanity check와 kernel smoke test

## 포함하지 않는 것

- NVIDIA Alpamayo model weights 또는 dataset files
- Alpamayo 전체 설치 가이드
- ncu 원본 report, 실패 실험 전체 archive, private 연구 로그
- speculative decoding 실험 결과
- `adaptive_flow`를 기본 성능 주장에 포함하는 근사 실행

## 확정 주장

- 모델 checkpoint 수정 없음
- 양자화 없음
- 기본 경로에서 근사 없음
- 동일 조건 locked-clock steady-state에서 eager 3,846 ms -> UMIC 2,701 ms
  (`-29.8%`, 19 decode steps, 2026-06-11 공식 벤치마크)
- 2026-07-06 VE 3종 fusion 추가 후 재검증: eager 3,185 ms -> UMIC 2,347 ms
  (`-26.3%`, 16-step 정규화), Vision Encoder 단독은 484 ms -> 194 ms (`-59.9%`).
  상세: [docs/260706_ve_production_integration.md](docs/260706_ve_production_integration.md)
- 출력 등가성 PASS: 3,106 tokens match, trajectory ADE 3.8 mm

## 아직 확장하면 좋은 검증

- 여러 clip과 long-tail scene에서 출력 등가성 표본 확대
- 보드 thermal history와 library version에 따른 expected range 보정
- private ncu 원자료 중 공개 가능한 최소 summary export
- speculative decoding은 별도 실험이 끝난 뒤 UMIC 이후 단계로 문서화
