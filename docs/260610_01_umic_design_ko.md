# UMIC: Unified-Memory Inference Compiler — iGPU 전용 컴파일 엔진 설계서

**날짜**: 2026-06-10
**선행 문서**: `260608_02_PyTorch_eager_커널융합_문제와_컴파일엔진_연구방향.md`, `260608_01_Alpamayo_4단계_실제_DRAM_대역폭_ncu_실측_분석.md`
**코드 위치**: `src/umic/` (스켈레톤), `scripts/compile_engine/` (Thor 검증 스크립트)

---

## 0. 한 문장 요약

**UMIC은 "범용 텐서 컴파일러"가 아니라, 실측 DRAM 트래픽(ncu)을 목적 함수로 삼아
transformer 계열 멀티스테이지 모델을 iGPU 통합 메모리 위에서 재컴파일하는
얇은(thin) 3계층 AOT 컴파일 레이어다.**

- 일반화: Alpamayo 전용이 아니라 **"Stage Pipeline + Transformer motif"로 표현 가능한 모든 모델** 대상
- 단순화: TVM/Inductor 같은 범용 컴파일러를 만들지 않는다. **닫힌 패턴 집합(~10개) + 정적 스케줄**만 다룬다
- 차별화: TRT-LLM이 못 보는 세 가지 — ① 스테이지 간 파이프라인, ② iGPU 통합 메모리, ③ 반복 구조(ODE 65-step, decode-until-EOS) — 를 1급 시민(first-class)으로 다룬다

---

## 1. 설계 원칙 (전문 엔지니어 관점에서의 의사결정)

### 원칙 1 — Bytes가 목적 함수다 (Measurement-Guided Compilation)

전통 컴파일러의 목적 함수는 FLOPs 또는 추정 latency다. UMIC의 목적 함수는
**"DRAM을 오간 byte 수"** 이며, 이는 ncu 하드웨어 카운터
(`lts__d_sectors_fill_sysmem.sum`)로 직접 검증 가능하다.

```
컴파일 결정 루프:
  1. Graph IR에서 각 커널 경계의 예상 DRAM read/write byte 계산 (정적 분석)
  2. 예측값을 ncu 실측값과 대조 → cost model 보정
  3. 패턴 fusion 후보를 "절약 byte" 순으로 정렬 → 큰 것부터 적용
  4. 적용 후 ncu 재측정 → 예측 대비 실제 절약 확인 (회귀 방지)
```

이것이 우리 연구의 novelty다: 컴파일러가 자기 결정을 **하드웨어 카운터로 닫힌 루프 검증**한다.
실측 데이터(VE 98GB, Prefill 232GB, Flow 244GB)가 이미 cost model의 ground truth로 존재한다.

### 원칙 2 — 두 체제(regime)에 두 가지 다른 무기

260609 실측이 보여준 핵심: 단계마다 병목 체제가 다르다.

| 체제 | 단계 | 증상 | 처방 |
|------|------|------|------|
| **A. Compute-limited** | VE (35%), Prefill (55%) | BW 여유 있음, activation 왕복 낭비 85×/15× | **커널 융합** — activation DRAM 왕복 제거 |
| **B. Memory-saturated** | Decode (89%), Flow (88%) | BW 포화, 가중치 읽기가 지배 | **융합 + 스케줄링** — Flow는 이론 대비 5.4× 낭비라 융합 여지 큼, Decode는 cross-stage prefetch와 launch overhead 제거만 |

> 양자화는 범위 밖 (연구 원칙). 체제 B에서 가중치 byte 자체는 못 줄인다 —
> 대신 "읽는 동안 다른 일을 시키는 것"(llm.npu bubble elimination)과
> "불필요하게 더 읽는 것 제거"(Flow 244→46GB)가 무기다.

### 원칙 3 — 닫힌 패턴 집합 (범용 fusion 탐색을 하지 않는다)

Transformer 계열 모델은 어떤 변형이든 아래 ~10개 motif의 조합이다.
범용 fusion 탐색(지수적 탐색 공간) 대신 **선언적 패턴 레지스트리**만 유지한다:

```
P1  norm_proj      : RMSNorm/LayerNorm + Linear(들)        ← Prefill ~15 GB 절약
P2  qkv_rope       : QKV proj + RoPE                       ← Prefill ~10 GB 절약
P3  sdpa           : FlashAttention (이미 융합됨, 통과)
P4  o_proj_residual: O proj + residual add
P5  gate_silu_mul  : gate/up proj + SiLU + elementwise mul ← Prefill ~30 GB 절약 (최대)
P6  down_residual  : down proj + residual add
P7  embed_head     : embedding / lm_head (통과 또는 fusion)
P8  adaln          : DiT AdaLN (scale/shift/gate)          ← Flow 전용
P9  ode_step       : x ← x + dt·v elementwise              ← Flow 전용
P10 patchify       : ViT patch embed + pos embed           ← VE 전용
```

새 모델 지원 = 패턴 추가 (엔진 수정 아님). 매칭 안 되는 서브그래프는 **eager로 폴백** —
TRT-LLM이 Alpamayo에서 실패한 이유(전체 그래프를 변환하지 못하면 전부 실패)를
폴백 우선 설계로 회피한다. **커버리지가 정확성을 막지 않는다.**

### 원칙 4 — iGPU 통합 메모리를 1급 자원으로 스케줄

discrete GPU 컴파일러가 모르는, Thor에만 있는 스케줄 가능한 자원:

```
자원 1: SM (단일 풀, 단일 컨텍스트)        ← compute
자원 2: DMA/copy engine                    ← cudaMemPrefetchAsync, SM과 독립
자원 3: GPU L2 32 MB                       ← cudaAccessPolicyWindow로 잔류 제어
자원 4: CPU (12 core) + CPU 캐시           ← 전처리/후처리, weight staging
자원 5: LPDDR5X 231 GB/s (공유 버스)       ← 모든 자원의 공통 제약 (예산)
```

스케줄러는 llm.npu의 bubble elimination 원리를 그대로 가져온다:
**"의존성을 지키면서 각 자원의 유휴 시간을 제거"**. 단, Thor 제약상
intra-layer SM 병렬화(Level 0)는 불가, inter-layer DMA↔SM 중첩(Level 1)과
inter-stage 중첩이 대상이다. BW 예산이 공유라는 점이 핵심 제약:
Decode(여유 ~25 GB/s) 중 prefetch는 layer 단위로는 불가능하고
stage 단위(Flow 가중치 4.56 GB를 decode 후반 step에 미리)로만 가능 — 실측이 이미 답을 줬다.

### 원칙 5 — 모든 것을 정적으로 (N=1, 고정 shape의 이점을 끝까지 사용)

10Hz 단일 추론, batch=1, 입력 shape 고정(6캠 영상, seq=3086, decode step shape는
AppendOnlyCache로 고정 가능, Flow 65 step 고정). 따라서:

- **정적 메모리 플랜**: 컴파일 시 arena 1개 할당, 모든 중간 텐서는 오프셋. 추론 중 malloc 0회
- **CUDA Graph per stage**: shape이 정적인 단계(decode step, flow step)는 그래프 캡처로
  launch overhead 제거 (Flow 48,232 커널 × ~3µs launch ≈ 145ms가 launch에만 소모될 수 있음)
  - ⚠️ 알려진 지뢰: VE의 `_deepstack_process` dynamic boolean indexing은 캡처 불가 (2026-05-28 확정)
    → stage별 선택 적용. 캡처 불가 stage는 stream 모드로 실행
- **in-place 버퍼 재사용**: Layer N 출력 버퍼 = Layer N+1 입력 (L2 잔류 확률 증가)

---

## 2. 아키텍처 — 3계층 IR (각 계층은 의도적으로 최소)

```
┌─────────────────────────────────────────────────────────────────┐
│  입력: 모델 (PyTorch nn.Module 그대로, 수정 없음)                  │
│       + 하드웨어 프로필 (thor.yaml: L2 32MB, 231GB/s, SM 11.0)    │
│       + ncu 실측 프로필 (단계별 GB, 커널 수, L2 hit)               │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌──── L2: Pipeline IR ────────────────────────────────────────────┐
│  Stage DAG + 반복 구조 + 가중치 footprint + 인터페이스             │
│                                                                  │
│  VE(1.15GB) → Prefill(15.2GB) → Decode(15.2GB, repeat≤19/EOS)   │
│                                → Flow(4.56GB, repeat=65)         │
│                                                                  │
│  패스: cross-stage prefetch 스케줄 생성 (BW 예산 시뮬레이션)        │
│        stage별 실행 모드 결정 (CUDA Graph / stream / persistent)  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌──── L1: Graph IR (stage별 torch.fx ATen 그래프) ─────────────────┐
│  패스 순서:                                                       │
│   1. capture     : torch.export / fx.symbolic_trace (stage 단위) │
│   2. canonicalize: DCE, dtype/layout 정규화, constant folding     │
│   3. analyze     : 커널 경계별 DRAM byte 정적 추정 → ncu 대조      │
│   4. fuse        : 패턴 레지스트리 매칭 → fused op로 치환          │
│                    (절약 byte 큰 순서로, 폴백 보장)                │
│   5. memplan     : liveness 분석 → arena 오프셋 배정, in-place    │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌──── L0: Kernel IR (커널 레지스트리) ─────────────────────────────┐
│  fused op → 구현 매핑:                                            │
│   - Triton 템플릿 (직접 작성, torch.compile 경유 안 함 ★)          │
│   - cuBLAS / SDPA / eager 폴백 (항상 존재)                        │
│   - persistent loop kernel (Flow ODE 65-step 내재화)              │
│  on-device 오토튜닝: 타일 크기를 Thor에서 실측 → 캐시(json)         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌──── Runtime (얇은 실행기) ───────────────────────────────────────┐
│  - arena 1회 할당, CUDA Graph replay, stream/event 의존성          │
│  - DMA prefetch 워커 (cudaMemPrefetchAsync, 별도 stream)          │
│  - L2 persistence window 설정/해제 (stage 전환 시)                 │
│  - 텔레메트리: stage별 시간 + (디버그 모드) ncu 연동                │
└──────────────────────────────────────────────────────────────────┘
```

★ torch.compile의 Inductor→Triton 경로는 Thor에서 사망 확인(2026-05-28).
**Triton 커널을 직접 작성**하면 Inductor의 깨진 API를 우회한다 — 단, Triton 3.7.0
런타임 자체가 SM 11.0에서 단독 동작하는지는 **M0에서 최우선 검증** (아래 §5).

### 왜 3계층인가 (그리고 왜 더 안 만드는가)

- **L2가 novelty**: TRT-LLM·Inductor·TVM 전부 단일 그래프만 본다. 스테이지 반복 구조와
  가중치 전환 타이밍은 L2에서만 보인다. 논문 contribution의 절반이 여기다.
- **L1은 빌리는 것**: torch.fx를 그대로 쓴다. IR을 새로 발명하지 않는다.
- **L0는 손으로 쓰는 것**: 패턴이 ~10개뿐이므로 커널 템플릿도 ~10개. 코드젠 안 한다.

이 절제가 "단순화된 구조"의 실체다. 전체 엔진 코어가 ~3,000줄을 넘지 않는 것을 목표로 한다.

---

## 3. 일반화 인터페이스 — Alpamayo는 설정일 뿐

사용자는 모델을 수정하지 않고 stage 경계만 선언한다:

```python
from umic import Stage, Pipeline, Repeat

pipe = Pipeline(
    name="alpamayo15",
    stages=[
        Stage("ve",      module=model.vision_encoder, weights_gb=1.153),
        Stage("prefill", module=model.lm,             weights_gb=15.168,
              mode="prefill"),
        Repeat(Stage("decode", module=model.lm, weights_gb=15.168,
                     mode="decode"),
               until="eos", max_iter=64),
        Repeat(Stage("flow",   module=model.action_expert, weights_gb=4.561),
               times=65),
    ],
    hw="configs/hw/thor.yaml",
    profile="profiling_results/260609_ncu_full/",   # 실측 → cost model 보정
)
engine = pipe.compile()        # AOT: capture → fuse → memplan → schedule
out = engine(camera, ego_hist) # 추론: malloc 0회, graph replay
```

다른 VLA/멀티모달 모델(예: OpenVLA, π0)도 같은 선언으로 올라간다.
**모델별 지식은 전부 패턴 레지스트리와 Pipeline 선언에 격리**되고, 엔진 코어는 모델 불가지론적이다.

---

## 4. 단계별 예상 효과 (실측 기반 추정)

| 단계 | 현재 (실측) | 적용 기술 | DRAM 목표 | 시간 추정 |
|------|-----------|----------|----------|----------|
| VE | 728ms, 98.1GB | P10 + P1~P6 fusion, CUDA Graph 부분 | ~10GB | ~400ms¹ |
| Prefill | 1,423ms, 232.0GB | P1+P2+P5 fusion (절약 ~55GB), memplan | ~120GB | ~900ms¹ |
| Decode | 1,503ms, 323.3GB | CUDA Graph(launch 제거), 이미 89% 포화 | ~310GB | ~1,400ms |
| Flow | 870ms, 244.3GB | persistent ODE kernel + P8/P9 fusion | ~60GB² | ~300ms² |
| 전환 bubble | 수십 ms | cross-stage DMA prefetch (decode 후반에 Flow 4.56GB) | — | ~0 |
| **합계** | **4,524ms** | | | **~3,000ms** |

> ¹ VE/Prefill은 compute-limited라 DRAM 절약이 시간 절약으로 1:1 환산되지 않음 —
> attention compute 하한이 별도로 존재. 보수적 추정치.
> ² Flow는 BW 88% 포화 상태이므로 byte 절약(244→60GB)이 거의 1:1로 시간 단축.
> **가장 확실한 1순위 타깃이 Flow다** (낭비 5.4×, BW 포화, 구조 단순·반복).

이 표의 숫자는 추정이며, 각 마일스톤에서 ncu로 검증한다 (원칙 1).

---

## 5. 구현 로드맵 (각 단계마다 "ncu 검증 게이트")

### M0 — 기반 검증 (1주) ★ 위험 요소 조기 격추
| 작업 | 검증 기준 |
|------|----------|
| Triton 3.7.0 단독(`@triton.jit` 직접) SM 11.0 동작 확인 | vector add + matmul이 Thor에서 정답 일치 |
| torch.fx로 LM 1개 layer / Action Expert 캡처 | 그래프 노드 수 확인, 캡처 불가 지점 목록화 |
| bytes-moved 정적 분석기 | 예측치가 ncu 실측의 ±30% 이내 (Prefill 기준) |

**M0 실패 시 플랜 B**: Triton 불가 → CUDA C++ 커널 (`torch.utils.cpp_extension`,
이미 PyTorch 소스 빌드 환경이 Thor에 있으므로 nvcc 13.0 사용 가능).
fx 캡처 불가 지점 → 해당 서브그래프 eager 폴백 (원칙 3).

### M1 — 체제 A 융합 (2~3주)
P5(gate_silu_mul) → P1(norm_proj) → P2(qkv_rope) 순서로 Triton 커널 작성·적용.
게이트: Prefill DRAM 232GB → 180GB 이하 (ncu), latency 회귀 없음, 수치 오차 < 1e-2 (bf16).

### M2 — Flow persistent kernel (3~4주) ★ 최대 기대 효과
65-step ODE 루프를 stage-persistent 실행으로: step 간 activation을 DRAM에 안 내리고
L2/shared에 유지, 가중치만 스트리밍. CUDA Graph로 launch overhead 제거 병행.
게이트: Flow DRAM 244GB → 100GB 이하, 870ms → 500ms 이하.

### M3 — Pipeline IR 스케줄러 (2주)
cross-stage prefetch (decode step 15부터 Flow 가중치 DMA, 실측상 3 step = 32ms 여유 확인됨),
stage별 L2 persistence window, CPU↔GPU 중첩 (다음 프레임 전처리).
게이트: stage 전환 bubble = 0 (Nsight Systems 타임라인으로 확인).

### M4 — 일반화 + 논문 (이후)
두 번째 모델(예: 표준 Qwen2-VL)을 Pipeline 선언만으로 올려 일반성 입증.
논문 프레임: *"Measurement-Guided Compilation for Multi-Stage VLA Inference on Unified-Memory Edge GPUs"*

---

## 6. 리스크 레지스터

| 리스크 | 확률 | 영향 | 완화 |
|--------|------|------|------|
| Triton 런타임이 SM 11.0 미지원 | 중 | 높음 | M0 최우선 검증, CUDA C++ 폴백 경로 확보 |
| fx 캡처가 Alpamayo 일부에서 실패 | 높음 | 중 | stage·layer 단위 부분 캡처 + eager 폴백 (설계에 내장) |
| Flow persistent kernel에서 activation이 shared/L2 초과 | 중 | 중 | step 단위 CUDA Graph replay로 후퇴 (launch 절감만으로도 이득) |
| fusion 후 bf16 수치 드리프트로 trajectory 품질 저하 | 낮 | 높음 | 매 커널 eager 대비 오차 게이트 + 최종 waypoint ADE 비교 |
| BW 공유 버스에서 prefetch가 compute를 방해 | 중 | 중 | BW 예산 시뮬레이터로 스케줄 사전 검증 (decode 중 layer prefetch 금지는 이미 실측 확정) |

---

## 7. M0 실측 결과 (2026-06-10, Thor에서 즉시 검증 완료)

스켈레톤 구현 직후 Thor에서 `scripts/compile_engine/260610_m0_smoke_test.py` 실행:

| 게이트 | 결과 | 의미 |
|--------|------|------|
| GATE1: Triton 3.7.0 단독 `@triton.jit` on SM 11.0 | ✅ **PASS** | **최대 리스크 해소.** Inductor 경로만 죽었고 Triton 런타임 자체는 Thor에서 정상 동작 — CUDA C++ 플랜 B 불필요 |
| GATE2: fused gate_silu_mul 수치 정확도 (bf16) | ✅ PASS (rel err 2.3e-3 < 1e-2) | 융합 커널 정확성 확인 |
| GATE3: prefill shape [3086,4096]×[4096,11008] 타이밍 | 초기 20.06ms → **튜닝 후 6.00ms** (eager 5.80ms) | 타일 스윕(BM=128,BN=128,BK=64,warps=8,stages=4)으로 cuBLAS 대비 +3%까지 도달. **단 eager 체인은 호출당 272MB DRAM 왕복 추가** — 시간 동률이어도 byte는 fused가 압도. ncu로 byte 검증이 다음 단계 |
| GATE4: Pipeline IR + prefetch 스케줄 dry-run | ✅ PASS | decode 중 Flow 4.56GB prefetch = 179.5ms @ 25GB/s headroom — decode 후반 ~3 step 이상 앞에서 시작해야 함을 스케줄러가 자동 도출 |

> GATE3 해석 주의: 단일 커널 wall-clock 비교는 DRAM 경합이 없는 마이크로벤치다.
> 실제 prefill(BW 55% 사용 중)에서는 272MB/layer 왕복 제거가 시간으로 환산된다.
> M1 게이트는 wall-clock이 아니라 **ncu 측정 byte 감소**다 (원칙 1).

## 8. M1 step 1 실측 결과 (2026-06-10, Thor ncu 검증 완료)

`src/umic/integrate.py`의 **구조 매칭 기반 무수정 주입**(duck-typed: `gate_proj`/`up_proj`/`down_proj` + SiLU를 가진 모든 모듈 — Qwen2/Qwen3/Llama/Alpamayo 2.0 무엇이든 모티프만 있으면 매칭)을 prefill shape의 표준 MLP에 적용하고 ncu로 측정:

| 항목 | eager (5 커널) | fused P5 (2 커널) | 변화 |
|------|--------------|------------------|------|
| DRAM read | 1,436.0 MB | 1,010.4 MB | −425.6 MB |
| DRAM write | 298.2 MB | 94.5 MB | **−203.7 MB** (= [3086,11008] 중간텐서 3개 × 68MB, 예측과 정확히 일치) |
| **총량** | **1,734.2 MB** | **1,105.0 MB** | **−629.2 MB (−36.3%)** |
| wall-clock (전체 MLP) | 12.11 ms | 10.43 ms | −13.9% |
| 수치 오차 (bf16) | — | rel 5.8e-3 | < 1e-2 게이트 통과 |

- **36 layer 환산: prefill에서 P5 하나로 ~22.6 GB 절약** (설계서 §4 예측 ~30 GB과 부합)
- 마이크로벤치에서도 wall-clock이 이미 13.9% 빨라짐 — DRAM 경합이 있는 실제 prefill에서는 더 커질 것
- fused read 1,010 MB는 여전히 이론(~364 MB)보다 큼 — 타일링에 의한 가중치 재읽기. L2-aware 타일 순서(다음 튜닝)로 추가 여지 있음
- 검증 스크립트: `scripts/compile_engine/260610_m1_mlp_ncu_test.py`, `260610_run_ncu_m1.sh`

### M1 step 2 — 실제 Alpamayo LM 주입, Prefill 전체 실측 (2026-06-10 완료)

`fuse_mlps(model.vlm.language_model)` 한 줄로 36개 LM MLP 패치 (체크포인트·모델 소스 무수정).
decode 보호: seq<64는 eager 디스패치 (`FUSE_MIN_ROWS`) — decode는 GEMV라 융합이 손해이기 때문.

| 항목 | eager 기준선 (260609 확정) | P5 fused | 변화 |
|------|--------------------------|----------|------|
| **Prefill DRAM 총량** | 231.97 GB | **148.12 GB** | **−83.85 GB (−36.2%)** |
| Prefill DRAM read | 179.90 GB | 104.26 GB | −75.6 GB |
| Prefill DRAM write | 52.07 GB | 43.86 GB | −8.2 GB |
| Prefill 커널 수 | 2,070 | 1,962 | −108 = 3커널/layer × 36 (예측 정확히 일치) |
| **Prefill wall-clock** | 1,423 ms | **1,030 ms** | **−393 ms (−27.6%)** |
| VE / Flow / Decode | 728 / 870 / 107ms·step | 727–754 / 891 / 106.7ms·step | 회귀 없음 (decode 폴백 작동 확인) |

**예측 초과 발견** (P5 단독 예측 −22.6 GB vs 실측 −83.9 GB):
※ 초기 해석("L2 경합 완화 → 주변 커널 연쇄 개선")은 2026-06-10 per-kernel class 분석으로
**기각** — 융합이 안 건드린 class(attention/reduce)의 byte 변화는 정확히 0이었다.
실제 출처: **eager cuBLAS gate/up GEMM의 병적 DRAM 비효율**. eager에서 gate/up GEMM 1개가
1.21 GB를 옮기고 있었고(이론 183 MB의 6.6×, L2 hit 63.4%), UMIC fused 커널은 같은 수학을
5.2× 적은 트래픽(469 MB/layer, L2 hit 93.7%)으로 수행. 중간텐서 제거 기여는 −13.7 GB,
**−70 GB는 GEMM 내부 가중치/타일 재읽기 제거**다. 추가 발견: 동일 nn.Linear가 실제 모델
내부에서 마이크로벤치 대비 2.3× 더 많은 DRAM을 소모 (cuBLAS 커널 선택의 컨텍스트 의존성)
→ 마이크로벤치 기반 결정 불가, measurement-guided 원칙의 직접 근거.
상세: `results/260610_m1_prefill/260610_l2_cascade_findings.md`

전체 파이프라인 환산: 4,838ms → **~4,445ms** (P5 한 패턴만으로 −8.1%).
스크립트: `scripts/compile_engine/260610_m1_prefill_e2e.py`, `260610_run_ncu_m1_prefill.sh`

다음 청크 후보: ① per-kernel L2 hit 비교로 연쇄 효과 검증, ② P1 norm_proj 커널, ③ Flow ODE fx 캡처 + 커널 경계 분석 (최대 기대 효과 구간).

## 9. TRT-LLM과의 관계 (재확인)

- LM Decode 경로는 TRT-LLM/표준 최적화가 이미 잘 푸는 영역이고 BW 89% 포화 — UMIC의 우선순위 아님
- UMIC의 전장은 TRT-LLM의 공백: **VE/Flow fusion, Flow persistent kernel, Pipeline IR 스케줄, iGPU 자원 모델**
- TRT-LLM 커널 아이디어(fused gate-SiLU, norm+proj)는 패턴 레지스트리에 차용하되, 구현은 Thor에서 직접 튜닝
