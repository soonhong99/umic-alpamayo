# 260611 공식 벤치마크 — 동일 조건 eager vs UMIC

**날짜**: 2026-06-11
**조건 (양쪽 동일)**: `jetson_clocks` 고정(GPU 1.386 GHz Min=Max), 같은 보드 세션에서 연속
실행, warmup 1회 + 측정 6회, steady-state = run 3+ 판정, 동일 입력(clip 030c760c, seq 3086).
**구성**: eager = `--fuse 0 --fuse-decode-cache 0 --fuse-graph 0` (모델 원본 그대로),
UMIC = 전체 융합(P5, q/o linear, RMSNorm, LayerNorm, vision/text RoPE) + InplaceKVCache
(decode contiguous + flow) + per-length decode CUDA Graph.

## 결과 (steady-state, 19-step run 기준)

| 단계 | eager | UMIC | 개선 |
|------|-------|------|------|
| VE | 532 ms | **305 ms** | **−42.7%** |
| LM Prefill | 1,090 ms | **588 ms** | **−46.1%** |
| LM Decode | 78.2 ms/step | **70.0 ms/step** | **−10.5%** |
| Flow | 721 ms | **449 ms** | **−37.7%** |
| **전체 (19 steps)** | **3,846 ms** | **2,701 ms** | **−29.8%** |
| 전체 (16 steps) | 3,617 ms | ~2,490 ms | −31.2% |

raw: eager runs 3-5 = wall 3,612/3,846/3,622 · UMIC runs 4-5 = wall 2,700/2,703.

## 기존 발표 수치의 정정

- ~~"eager 4,838ms 대비 −48.7%"~~ → **−29.8%가 공식 수치.** 구 기준선 4,838ms는
  기본 거버너 + 2-run(cold) 조건이라 환경 패널티가 포함돼 있었다. eager도 클럭 고정 +
  장기 웜업으로 3,846ms까지 내려온다.
- 어제 보고한 단계별 상대 개선율(A/B)들은 **유효** — 양쪽 모두 같은(거버너) 조건에서
  측정했고, 상대비는 조건 간 보존됨이 확인됨 (decode: 거버너 0.91 vs 고정 0.90).

## 귀속(attribution)의 수정

- **decode**: 클럭 고정+웜업 조건에서 eager(DynamicCache)가 이미 78.2ms — cat-copy
  비용이 이 조건에선 미미. UMIC의 78→70은 주로 **norm/RoPE 융합 + CUDA Graph**의 기여.
  InplaceKVCache의 decode 기여는 거버너 조건에서 컸던 것(107→97)과 달리 고정 조건에선 작음.
  (flow의 InplaceKVCache 기여는 byte 자체를 줄이므로 조건 무관하게 유효 — flow −37.7%)
- VE/Prefill/Flow 개선은 ncu byte 감소(−48/−66/−31%)와 정합 — 융합의 기여.

## 남은 검증 항목

1. **출력 등가성**: waypoint ADE(eager vs UMIC) 미측정 — 다음 작업
2. UMIC 16-step steady 표본이 적음 (graph 캡처와 섞임) — 보고서 전 추가 표본 권장
