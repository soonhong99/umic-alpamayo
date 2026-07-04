# UMIC 온보딩 체크리스트

새 연구생이 Thor 보드에서 이 repo를 처음 실행할 때의 최소 경로다. Alpamayo 설치와 HF
캐시는 이미 준비되어 있다고 가정한다.

## 1. 한 번에 실행

```bash
git clone https://github.com/soonhong99/umic-alpamayo.git
cd umic-alpamayo
bash scripts/run_all.sh
```

`run_all.sh`는 다음 순서로 진행한다.

1. `sudo jetson_clocks`로 GPU 클럭을 고정한다.
2. `~/alpamayo1.5/a1_5_venv`가 있으면 활성화한다.
3. `scripts/check_env.py`로 Python, CUDA, Triton, 커널 smoke test를 확인한다.
4. Alpamayo를 로드해 eager와 UMIC을 같은 조건에서 비교한다.

## 2. 단계별 실행

환경만 먼저 확인하려면:

```bash
bash scripts/setup_thor.sh
```

벤치마크만 다시 돌리려면:

```bash
python scripts/run_pipeline.py --mode both
python scripts/run_pipeline.py --mode umic --runs 6 --warmup 8
```

기존 코드에서 API로 쓰려면 editable install 후 import한다.

```bash
python -m pip install -e .
```

```python
import umic

report = umic.apply(model)
```

## 3. 자주 막히는 지점

| 증상 | 의미 | 조치 |
|------|------|------|
| `alpamayo1_5 package NOT importable` | Alpamayo venv가 활성화되지 않음 | `source ~/alpamayo1.5/a1_5_venv/bin/activate` 후 재실행 |
| `model HF cache missing` | 모델 weight가 로컬 HF cache에 없음 | NVIDIA/HF access를 확인하고 보드에서 한 번 다운로드 |
| `GPU clock locked` 실패 | DVFS governor 상태 | `sudo jetson_clocks` 후 재실행 |
| `triton import` 실패 | fused kernel 실행 불가 | Thor venv의 Triton 3.7.0 설치 상태 확인 |
| `[SLOW]` 판정 | 대개 clock 미고정 또는 warmup 부족 | `run_all.sh` 기본값(warmup 5)을 사용 |
| `ModuleNotFoundError: yaml` | PyYAML 미설치 | `python -m pip install -e .` 또는 `python -m pip install PyYAML` |

## 4. 해석 기준

- 공식 기준은 `docs/260611_official_benchmark.md`의 2026-06-11 locked-clock steady-state
  A/B 결과다.
- `configs/expected_thor.yaml`의 범위는 같은 보드에서 정상 실행인지 판단하기 위한 sanity
  range다.
- 출력 등가성은 `docs/260611_output_equivalence.md` 기준으로 PASS 상태다.
- speculative decoding은 이 repo의 범위가 아니며, 실험 완료 후 별도 문서 또는 repo에서
  다룬다.
