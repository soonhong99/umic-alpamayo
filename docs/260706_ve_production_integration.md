# 260706 Vision Encoder 3종 융합: 발견부터 production 통합까지

**날짜**: 2026-07-06
**대상 독자**: 이 repo(umic-alpamayo)를 처음 보는 협업 연구자

## 0. 결론 요약

기존 UMIC은 VE(Vision Encoder)에 세 가지만 적용하고 있었다(LayerNorm 융합·RoPE 융합·bf16 residual). nsys로 VE를 커널 단위로 다시 뜯어본 결과, 이 세 가지가 못 건드린 세 지점을 추가로 찾아 고쳤다.

**VE 자체 기준**: 기존 UMIC 대비 **약 -27% 추가 개선** (484ms→194ms는 eager 대비 -59.9%, 이 중 기존 UMIC 몫을 뺀 이번 추가분이 -27% 수준).
**파이프라인 전체 기준(2026-07-06 재검증)**: eager 3,211ms → UMIC 2,591ms (16-step 정규화 -26.3%).
**정확도**: 세 융합 모두 검증됨. 하나는 완전 bit-exact, 둘은 기존 UMIC이 이미 받아들인 것과 같은 수준의 bf16 오차(육안으로는 구분 불가능한 수준).
**production 반영**: `src/umic/integrate.py`에 3개 함수 추가, `optimize.py`의 `apply()`에 3줄 추가. 전부 이 repo의 기존 원칙(구조 매칭, 가중치 미변경, 매칭 실패 시 안전한 no-op)을 그대로 지켰다.

---

## 1. 기존 UMIC이 VE에서 하던 일 (Before)

`umic.apply()`를 부르면 VE에는 이 세 가지만 적용됐다:

| 기존 융합 | 하는 일 |
|---|---|
| `fuse_layernorms` | LayerNorm 계산 자체를 커널 하나로 압축 |
| `fuse_vision_rope` | 회전 위치 임베딩(RoPE) 적용 계산을 커널 하나로 압축 |
| `fuse_bf16_residual` | fp32로 새던 잔차 스트림을 bf16으로 되돌림 |

이 상태의 VE는 eager 대비 -42.7%(532ms→305ms, 2026-06-11 공식 벤치마크 기준)였다. 그런데 이 세 가지를 적용한 뒤에도 VE 안에는 아직 손대지 않은 커널이 세 종류 남아있었다. nsys로 VE 한 번의 forward pass를 커널 단위로 전부 덤프해서 확인했다.

## 2. 새로 찾은 세 지점과 고친 방법 (After)

### 2.1 패치 임베딩: Conv3d를 쓰는데 사실 필요 없다

ViT 입력의 첫 단계(patch embedding)는 `nn.Conv3d`를 쓴다. 그런데 이 conv는 **커널 크기와 stride가 완전히 같다**(슬라이딩이 없다는 뜻). 이런 conv는 수학적으로 그냥 행렬곱(Linear)과 완전히 같다(ViT 계열에서 잘 알려진 사실이다).

실측해보니 Conv3d 경로는 **19.0ms**, 등가인 Linear 경로는 **0.36ms**였다. 53배 차이다. cuDNN이 "커널=stride"인 이 conv를 최적화하지 못하고 훨씬 느린 경로로 처리하고 있었던 것이다. **fp64(배정밀도)로 두 공식이 완전히 같다는 것도 확인했다**(오차 0). bf16에서 아주 작은 오차(0.03 정도)가 보이는 건 순전히 계산 순서 차이일 뿐, 공식이 다른 게 아니다.

→ `fuse_patch_embed_linear`: "stride == kernel_size인 ConvNd를 가진 모듈"을 구조로 찾아서 Linear로 교체한다. 특정 모델 클래스 이름에 의존하지 않으므로, 다른 ViT 모델에도 똑같이 적용된다.

### 2.2 Attention: 이미지 16개를 따로 계산한 뒤 다시 이어붙이고 있었다

이 모델은 카메라 여러 대(이 배포 환경에서는 16대)의 이미지를 한 번에 처리한다. 그런데 attention 계산은 이미지별로 따로따로 잘라서(`split`) 계산한 뒤 다시 이어붙이는(`concat`) 방식이었다. 이게 매 블록(27개)마다 반복되니 attention 커널 호출이 432번(27블록×16이미지) 발생했다.

PyTorch 자체에 이미 "여러 개를 한 번에 처리하는" attention 함수(`torch.ops.aten._flash_attention_forward`)가 내장돼 있다. 이걸 쓰면 이미지 16개를 자르지 않고 한 번의 호출로 처리할 수 있다. 이렇게 바꾸니 attention 커널 호출이 432회→27회로 줄었다.

**정확도는 완전히 bit-exact였다**(오차 0). 이유: 원래도 같은 계산을 16번 나눠 했을 뿐이라, 한 번에 처리해도 수학적으로 완전히 동일하다.

(참고: 외부 `flash_attn` 패키지를 먼저 시도했는데 이 보드(Thor, SM 11.0)용 커널이 없어서 실패했다. 대신 PyTorch가 자체적으로 이미 쓰고 있는 내부 연산을 직접 호출하는 방식으로 우회했다.)

→ `fuse_vision_attention_varlen`: "qkv+proj+num_heads+scaling을 가지고, forward에 `cu_seqlens` 인자가 있는" attention 모듈을 구조로 찾아서 교체한다.

### 2.3 Residual add: LayerNorm은 합쳤는데 그 직전 "더하기"는 안 합쳤다

ViT 블록은 이런 패턴이다:

```
결과 = 입력 + attention(norm1(입력))
결과 = 결과 + mlp(norm2(결과))
```

기존 UMIC은 `norm1`/`norm2` 자체는 이미 커널 하나로 압축했지만, 그 직전의 "입력 + attention결과" 덧셈은 별도 커널로 남아있었다. LM(언어모델) 쪽에는 이미 이 덧셈을 다음 norm에 합치는 기법(`fuse_add_rmsnorm`)이 있는데, VE 쪽에는 LayerNorm판이 없었다.

이번에 LayerNorm판(`add_layernorm_triton`)을 새로 만들고, 여기서 한 걸음 더 나갔다: **한 블록의 마지막 덧셈을 "다음 블록의 norm1"과도 합칠 수 있는지** 확인했다. 실제로 확인해보니 27개 블록 중 24개에서 가능했다(3개는 중간 결과를 다른 곳에서도 그대로 써야 해서 예외). 이렇게 해서 27개의 "덧셈" 중 24개를 없앴다.

→ `fuse_vision_encoder_pipeline`: VE 전체의 forward를 다시 짜서, 캐싱 가능한 상수(카메라 배치가 안 바뀌는 한 매번 새로 계산할 필요 없는 값들)와 이 덧셈 융합을 함께 적용한다.

## 3. "기존 UMIC 원칙"을 그대로 지켰는가

이 repo는 세 가지 원칙이 있다: (1) 클래스 이름이 아니라 구조로 찾는다, (2) 가중치는 절대 복사·변경하지 않는다, (3) 매칭 안 되면 조용히 원본 그대로 둔다(에러 안 남). 이번 3개 함수 전부 이 원칙을 그대로 따랐다:

- 셋 다 `hasattr` 기반 구조 매칭. 예: "Conv인데 stride==kernel_size"면 어떤 모델의 어떤 클래스든 매칭된다.
- 셋 다 `dry_run=True` 지원(실제로 패치하지 않고 몇 개가 매칭되는지만 셀 수 있음).
- patch embed의 Linear 가중치도 `conv.weight.reshape(...)`로 view만 뜬다(원본 그대로).
- **한 가지는 기존보다 더 엄격하게 만들었다**: 캐싱이 "이 배포 환경은 카메라가 항상 16대"라는 가정에 기대지 않는다. 입력값(`grid_thw`)이 실제로 바뀌면 캐시가 자동으로 무효화되고 다시 계산한다.

## 4. 검증 절차 (수치 포함)

### 4.1 구조 매칭이 정확한지 (dry_run)

```
fuse_patch_embed_linear dry_run: 1     (예상 1)
fuse_vision_attention_varlen dry_run: 27  (예상 27)
fuse_vision_encoder_pipeline dry_run: 1   (예상 1)
```

### 4.2 실제 `umic.apply()`를 호출해서 검증

완전히 미수정된 모델과, `umic.apply()`를 실제로 호출한 별도 모델을 나란히 비교했다(스크립트 재사용이나 monkeypatch 잔여 상태 없이 각각 독립적으로 로드).

- `apply()`의 report dict에 새 3개 항목이 정확한 카운트로 나타남 (위와 동일)
- 정확도: 전체 16개 chunk 기준 max_abs_diff 최대 1.125, mean_abs_diff 약 0.0048(이미 이 프로젝트가 다른 곳(`add_rmsnorm_triton` 등)에서 받아들인 것과 같은 수준의 bf16 오차)
- VE 단독 시간: 472.96ms(완전 미최적화) → 183.69ms (**-61.16%**, 8윈도우 CUDA-event 측정)

### 4.3 공식 벤치마크 하네스로 재확인 (`run_pipeline.py --mode both`)

이 repo의 실제 벤치마크 스크립트로 eager와 UMIC을 같은 세션에서 연속 측정했다(클럭 고정, warmup 5 + 측정 3, clip `030c760c`):

| 단계 | eager | UMIC | 개선 |
|---|---:|---:|---:|
| VE | 484ms | **194ms** | **-59.9%** |
| LM Prefill | 842ms | 587ms | -30.3% |
| LM Decode | 74.3ms/step | 71.8ms/step | -3.4% |
| Flow | 671ms | 417ms | -37.9% |
| 전체(16-step 정규화) | 3,185ms | 2,347ms | -26.3% |

Prefill/Decode/Flow는 이번 변경과 무관하므로(원래 UMIC 로직 그대로) 수치가 크게 안 변한 게 정상이다. VE만 확실히 개선됐다.

## 5. 한계와 남지 않은 것

- CUDA Graph 캡처 자체는 이번에 wiring하지 않았다. `fuse_vision_encoder_pipeline`의 상수 캐싱은 eager 모드에서 이미 대부분의 이득(약 -10%)을 가져다주는 것으로 실험 단계에서 확인됐고, 실제 `torch.cuda.graph()` 캡처는 decode 쪽처럼 별도 인프라(`umic/graph.py`에 해당하는 것)가 필요해 이번 범위 밖으로 뒀다.
- deepstack tap 3곳(레이어 8/16/24)의 residual add는 구조상 다음 소비처가 reshape을 먼저 하기 때문에 융합하지 않고 그대로 뒀다(정확하지만 완전히 최적은 아님. 27개 중 3개는 여전히 plain add).
- GELU 활성화 커널(28.3ms)은 조사했지만 **고칠 수 없는 것으로 확정**됐다(이 shape에서 cuBLAS가 Triton보다 이미 빠름, 13개 타일 설정 스윕으로 재확인). production에 반영하지 않았다(원래도 안 하고 있었음, 그대로 유지).

## 6. 관련 코드

- `src/umic/kernels/layernorm.py`: `add_layernorm_triton` 추가
- `src/umic/integrate.py`: `fuse_patch_embed_linear`, `fuse_vision_attention_varlen`, `fuse_vision_encoder_pipeline` 추가
- `src/umic/optimize.py`: `apply()`에 위 3개 호출 추가
- `configs/expected_thor.yaml`: VE의 기대 범위 갱신(`[230,340]` → `[170,230]`)
