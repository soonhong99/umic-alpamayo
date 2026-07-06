# Changelog

이 파일은 umic-alpamayo의 날짜별 변경 이력을 기록한다. 최신 항목이 위에 온다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 관례(추가/변경/수정 구분)를
따르되, 한 눈에 보기 쉽도록 아래 개정이력 표를 함께 둔다.

**README는 항상 "지금 검증된 최신 수치"만 보여준다.** 과거 측정값과 그 사유는 이 파일에서
찾는다.

## 개정이력 요약

| 버전 | 개정일자 | 개정내용 | 관련 커밋 |
|---|---|---|---|
| 0.3.0 | 2026-07-06 | VE 3종 fusion 추가 (patch embed·varlen attention·residual+LN 파이프라인) | [`6ca14de`](https://github.com/soonhong99/umic-alpamayo/commit/6ca14de) |
| 0.2.0 | 2026-07-05 | 온보딩/scope 문서 정리, LICENSE 추가 | [`df5b3db`](https://github.com/soonhong99/umic-alpamayo/commit/df5b3db) |
| 0.1.0 | 2026-07-04 | 최초 공개: UMIC 런타임 8종 최적화 + 벤치마크 하네스 | [`2dcb969`](https://github.com/soonhong99/umic-alpamayo/commit/2dcb969)–[`dffb6c7`](https://github.com/soonhong99/umic-alpamayo/commit/dffb6c7) |

---

## 2026-07-06: VE 3종 fusion 추가

### 배경

nsys로 Vision Encoder를 커널 단위로 재해부한 결과, 기존 3종 융합(LayerNorm·RoPE·bf16
residual)이 못 건드린 세 지점(패치 임베딩·attention split/concat·residual add)을 확인했다.
상세 조사 과정: [docs/260706_ve_production_integration.md](docs/260706_ve_production_integration.md).

### Added

- `fuse_patch_embed_linear`: stride==kernel_size인 ConvNd(패치 임베딩)를 등가 Linear로 교체.
- `fuse_vision_attention_varlen`: 이미지별 split+attn+concat을 `torch.ops.aten._flash_attention_forward`
  기반 단일 packed varlen attention 호출로 교체.
- `fuse_vision_encoder_pipeline`: grid_thw 파생 상수의 자기무효화 캐시 + 블록 내부·블록 간
  residual add를 다음 LayerNorm에 융합(27개 중 24개).
- `add_layernorm_triton`(`kernels/layernorm.py`): `add_rmsnorm_triton`의 LayerNorm판.

### Changed

- VE: 484ms → 194ms (**-59.9%**, eager 대비).
- 파이프라인 전체: 3,185ms → 2,347ms (**-26.3%**, 16-step 정규화, `run_pipeline.py --mode both`
  동일 세션 연속 측정, 클럭 고정, clip `030c760c`).
- `configs/expected_thor.yaml`의 VE 기대 범위: `[230,340]` → `[170,230]`.
- `docs/current_scope.md`의 "채택된 UMIC 최적화": 8종 → 11종.

### 검증

- 구조 매칭(dry_run): patch embed 1개, attention 27개, pipeline 1개, 전부 예상과 일치.
- 정확도: `umic.apply()` 실제 호출 vs 완전 미수정 baseline, max_abs_diff 1.125,
  mean_abs_diff ~0.0048 (기존 `add_rmsnorm_triton` 융합과 같은 수준의 bf16 오차).
- patch embed는 fp64로 공식 자체가 완전히 동일함을 별도 확인(오차 0).
- attention varlen 융합은 완전 bit-exact(오차 0, 같은 커널을 배치 방식만 바꿔 호출).

---

## 2026-07-05: 온보딩/scope 문서 정리

### Added

- `LICENSE`
- `docs/current_scope.md`: 이 repo에 포함/제외되는 범위 명시.
- `docs/onboarding.md`: 새 연구생용 첫 실행 체크리스트.

### Changed

- README에 scope/onboarding 링크 추가, editable install(`pip install -e .`) 안내 추가.
- `src/umic/kernels/linear.py` 독스트링에서 private 연구 repo 경로 참조 제거(공개 저장소 기준으로 일반화).

---

## 2026-07-04: 최초 공개, UMIC 런타임 8종 최적화

### Added

- `umic.apply(model)` 원콜 API와 `UmicConfig`.
- 채택된 최적화 8종: MLP gate/SiLU/up fusion, q/o projection Triton dispatch, RMSNorm fusion,
  residual add + norm fusion(LM), text/vision RoPE fusion, ViT LayerNorm fusion,
  InplaceKVCache, per-KV-length decode CUDA Graph.
- 단계별 CUDA-event 타이밍 하네스(`bench.py`)와 기대범위 판정(`configs/expected_thor.yaml`).
- `scripts/run_all.sh` / `run_pipeline.py` / `check_env.py` / `setup_thor.sh`.
- 공식 벤치마크·출력 등가성 문서.

### 측정

- 2026-06-11 원본 연구 repo에서 측정(이 repo로 이식): eager 3,846ms → UMIC 2,701ms
  (**-29.8%**, 19 decode steps, 클럭 고정 steady-state). 상세:
  [docs/260611_official_benchmark.md](docs/260611_official_benchmark.md).
