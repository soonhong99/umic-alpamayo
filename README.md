# umic-alpamayo

**UMIC (Unified-Memory Inference Compiler) 런타임: Jetson AGX Thor에서 NVIDIA Alpamayo 1.5 추론을 모델 무수정으로 가속.**

- 체크포인트 수정 없음, 양자화 없음: 같은 수학, 실행 스케줄만 교체
- 공식 벤치마크 (2026-06-11, 동일 조건): eager 3,846 ms → **UMIC 2,701 ms (전체 −29.8%)**
- 출력 등가성 검증 완료: 3,106 토큰 전부 일치, 궤적 ADE 3.8 mm ([docs/260611_output_equivalence.md](docs/260611_output_equivalence.md))

| 단계 | eager | UMIC | 개선 |
|------|-------|------|------|
| Vision Encoder | 532 ms | **305 ms** | −42.7% |
| LM Prefill | 1,090 ms | **588 ms** | −46.1% |
| LM Decode | 78.2 ms/step | **70.0 ms/step** | −10.5% |
| Flow (Action Expert) | 721 ms | **449 ms** | −37.7% |
| **전체 (19 decode steps)** | **3,846 ms** | **2,701 ms** | **−29.8%** |

---

## 1. 요구 환경

| 항목 | 값 |
|------|-----|
| 보드 | Jetson AGX Thor (SM 11.0, JetPack 7, CUDA 13.0) |
| Python | 3.10+ (Thor 표준: 3.12) |
| PyTorch | 2.8.0 (Thor는 소스 빌드 필수: 공식 aarch64+CUDA13 wheel 없음) |
| Triton | 3.7.0 (직접 `@triton.jit`은 SM 11.0에서 정상 동작; 없으면 전부 eager로 폴백) |
| transformers | ≥ 4.56 (`Cache.layers` API 기준) |
| Alpamayo | `nvidia/Alpamayo-1.5-10B` HF 캐시 (선택: 없어도 커널 검증까지는 가능, §5) |

Thor의 기존 Alpamayo venv(`~/alpamayo1.5/a1_5_venv/`)를 그대로 쓰면 위 조건이 모두 충족된다.

## 2. 빠른 시작 (Thor 보드에서)

```bash
git clone https://github.com/soonhong99/umic-alpamayo.git
cd umic-alpamayo
bash scripts/run_all.sh             # 초기 세팅부터 벤치마크까지 전부 한 번에
```

`run_all.sh` 하나가 실험 초기 세팅 전체를 순서대로 수행한다:
① `sudo jetson_clocks` 클럭 고정 (비밀번호 1회 입력, §7 규칙 1) → ② `~/alpamayo1.5/a1_5_venv` 활성화 → ③ 환경+커널 점검 (실패 시 벤치마크 진입 전에 중단) → ④ eager vs UMIC 벤치마크 (**warmup 5회** + 측정 3회, §7 규칙 2 기본 내장).

추가 인자는 그대로 벤치마크에 전달된다:

```bash
bash scripts/run_all.sh --mode umic            # UMIC만 측정
bash scripts/run_all.sh --runs 6 --warmup 8    # 더 긴 측정
```

단계를 나눠 실행하려면: `bash scripts/setup_thor.sh` (세팅+점검만) 후 `python scripts/run_pipeline.py --mode both`. `run_pipeline.py`를 단독 실행해도 클럭 미고정을 감지하면 스스로 `sudo -n jetson_clocks`로 고정을 시도하고, 실패하면 경고를 출력한다.

## 3. 실행하면 무엇이 나오는가

run마다 단계별 ms가 이 보드의 기대 범위([configs/expected_thor.yaml](configs/expected_thor.yaml))와 함께 출력되고, 범위 안이면 `[OK]`, 벗어나면 `[SLOW]`/`[FAST]`로 판정된다.

아래는 2026-07-04 이 repo 검증 실행의 실제 출력이다:

```
=== umic run 3/3 (19 decode steps) ===
stage                measured      expected     verdict
-------------------------------------------------------
VE                    259.0 ms       230-340    [OK]
LM Prefill            577.0 ms       520-660    [OK]
Decode/step (SS)       70.5 ms         64-78    [OK]
Flow                  412.8 ms       370-500    [OK]
Wall total           2614.9 ms     2300-2950    [OK]

 eager median: VE 482 | Prefill 838 | Decode 73.1/step | Flow 666 | wall 3177 ms
  umic median: VE 259 | Prefill 577 | Decode 70.5/step | Flow 413 | wall 2615 ms
UMIC vs eager wall (16-step normalized): -24.7%  (3155 -> 2377 ms; official reference: -29.8%)
```

판정 해석:
- `[SLOW]`: 거의 항상 클럭 미고정(`sudo jetson_clocks` 후 재실행) 또는 웜업 부족(steady state는 warmup 포함 5+ run 뒤, 기본값이 warmup 5). **첫 UMIC run은 CUDA Graph 캡처(~19개)로 decode가 ~100 ms/step 나오는 것이 정상**이며, 판정은 run 2+를 기준으로 본다.
- `[FAST]`: 다른 clip이거나 decode step 수가 짧은 경우(wall은 step 수에 비례).
- decode step 수는 샘플링에 따라 run마다 13~20으로 달라지므로, eager vs UMIC 최종 비교는 **16-step 정규화 wall** 기준으로 출력된다.
- 전체 개선율은 그날의 eager 기준선에 따라 −18 ~ −30% 범위로 움직인다 (UMIC 절대치는 안정적, eager가 보드 상태에 더 민감). 공식 기준 수치는 [docs/260611_official_benchmark.md](docs/260611_official_benchmark.md).

결과 JSON은 `results/run_<timestamp>.json`에 저장된다 (run별 전체 수치 + median 요약).

## 4. 코드에서 직접 쓰기 (API)

벤치마크 없이 기존 추론 코드에 UMIC만 얹으려면 한 줄이면 된다:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
import torch, umic

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, local_files_only=True,
).cuda().eval()

report = umic.apply(model)      # 전체 채택 세트 적용, 패치 카운트 리포트 반환
```

- 매칭은 전부 구조 기반(duck-typing)이라 클래스명이 바뀌어도 모티프(gate/up/down SiLU MLP, RMSNorm, pre-norm decoder layer 등)가 유지되면 그대로 동작한다. 매칭 실패는 에러가 아니라 해당 융합만 no-op.
- 선택 적용: `umic.apply(model, umic.UmicConfig(decode_graph=False))` 처럼 항목별 on/off. 항목 정의는 [src/umic/optimize.py](src/umic/optimize.py)의 `UmicConfig` docstring 참고.
- `adaptive_flow=True`는 유일한 **근사** 옵션(flow −40%, 궤적 ~4 cm 편차)이라 기본 off.

## 5. Alpamayo가 아직 없는 보드라면

모델 없이도 엔진 자체는 검증할 수 있다:

```bash
python scripts/check_env.py
```

torch/CUDA/Triton/클럭 상태를 점검한 뒤, 모든 UMIC Triton 커널을 실제 파이프라인 shape의 랜덤 텐서로 eager 참조 구현과 비교한다(정확도 + 커널별 ms). 전부 `[OK]`면 엔진은 준비된 것이고, Alpamayo 설치 후 `run_pipeline.py`만 실행하면 된다.

Alpamayo 설치는 NVIDIA 공식 절차를 따른다 (HF `nvidia/Alpamayo-1.5-10B`, gated license 동의 필요). 로딩 규칙 주의: `Alpamayo1_5.from_pretrained`는 로컬 절대경로를 받지 못한다. 반드시 HF repo id + `local_files_only=True`.

## 6. 프로젝트 구조

```
src/umic/
  optimize.py     umic.apply() 원콜 API + UmicConfig
  integrate.py    구조 매칭 융합 주입 (fuse_mlps / fuse_rmsnorms / ...)
  kernels/        Triton 커널 5종 (fused_ffn, linear, rmsnorm, layernorm, rope)
                  전부 eager 폴백 내장
  cache.py        InplaceKVCache (append-then-crop 루프의 cat-copy 제거)
  graph.py        per-KV-length decode CUDA Graph (dispatch bubble 제거)
  diffusion.py    adaptive flow (opt-in 근사)
  bench.py        단계별 CUDA-event 타이밍 하네스 + 기대범위 판정
scripts/
  run_pipeline.py 메인 실행 (--mode umic|eager|both)
  check_env.py    환경 점검 + 커널 스모크 (Alpamayo 불필요)
  setup_thor.sh   클럭 고정 + venv + 점검 원커맨드
configs/
  expected_thor.yaml  이 보드의 단계별 기대 ms 범위 (판정 기준)
docs/               공식 벤치마크 / 출력 등가성 / 설계서
```

## 7. 측정 규칙 (지키지 않으면 수치가 어긋난다)

1. **측정 전 `sudo jetson_clocks` 필수.** decode처럼 memory-bound인 단계는 SM 사용률이 낮아 DVFS 거버너가 클럭을 올리지 않는다. 같은 코드가 거버너 상태에서 ~107 ms/step, 고정 상태에서 70 ms/step. `run_all.sh`가 첫 단계로 수행하며, `run_pipeline.py` 단독 실행 시에도 자동 감지+고정 시도.
2. **steady state는 warmup 포함 5+ run 후 판정.** 클럭 고정 상태에서도 allocator/페이지 워밍으로 모든 단계가 run 0→4에 걸쳐 계단식으로 내려온다 (실측: VE 427→305 ms, decode 102→70 ms/step). `run_pipeline.py` 기본값이 **warmup 5 + 측정 3**이라 측정 run은 전부 steady state에서 시작한다.
3. **첫 UMIC 추론은 CUDA Graph 캡처 비용이 섞인다.** 10 Hz 연속 운영을 모사한 설계라 두 번째 추론부터는 순수 replay다.

## 8. 배경 문서

- [docs/260611_official_benchmark.md](docs/260611_official_benchmark.md): 공식 수치의 측정 조건과 구 수치 정정 이력
- [docs/260611_output_equivalence.md](docs/260611_output_equivalence.md): 출력 등가성 게이트 (토큰 일치 + ADE 3.8 mm)
- [docs/260610_01_umic_design_ko.md](docs/260610_01_umic_design_ko.md): UMIC 설계서 (3계층 IR, measurement-guided compilation)
- 실험 전 과정(ncu 측정, 기각된 시도 포함)은 연구 repo [soonhong99/umic](https://github.com/soonhong99/umic) (private)

License: research use only. Alpamayo 1.5는 NVIDIA non-commercial research license를 따른다.
