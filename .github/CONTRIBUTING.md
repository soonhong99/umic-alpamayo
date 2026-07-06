# 기여 가이드

이 repo는 Jetson AGX Thor 실물 보드가 있어야 대부분의 코드를 실행·검증할 수 있는 연구
runtime repo다. 아래 순서를 지키면 리뷰가 빨라진다.

## 1. 시작하기 전에

- 온보딩 체크리스트: [docs/onboarding.md](../docs/onboarding.md)
- 이 repo에 포함/제외되는 범위: [docs/current_scope.md](../docs/current_scope.md)
- 새 최적화를 추가하려는 경우, 이미 시도했다가 기각된 것들을 먼저 확인한다
  ([README.md](../README.md) §2의 "기각 목록").

## 2. 이슈

- 버그 리포트, 새 최적화 제안 모두 이슈 템플릿을 쓴다(이슈 생성 시 자동으로 뜬다).
- Thor 보드에서만 재현되는 문제는 `jetson_clocks` 고정 여부와
  `python scripts/check_env.py` 출력을 함께 첨부한다.

## 3. 새 최적화(fusion)를 추가할 때

이 repo의 원칙([README.md](../README.md) §2)을 반드시 지킨다.

1. **목적함수는 DRAM bytes.** ncu로 실측해서 이론값 대비 초과하는 지점만 후보로 삼는다.
2. **매칭은 클래스가 아니라 구조.** `hasattr` 기반으로 모티프를 찾는다(`integrate.py`의
   기존 `_is_*` 함수들을 참고). 매칭 실패는 에러가 아니라 조용한 no-op이어야 한다.
3. **채택은 실파이프라인 실측으로만.** 단독 커널 벤치에서 이겨도 `run_pipeline.py --mode both`
   e2e에서 손해면 기각한다.
4. **가중치는 절대 복사·변경하지 않는다.** view/reshape만 쓴다.
5. `dry_run=True` 파라미터를 지원해서 실제 패치 없이 매칭 개수를 셀 수 있게 한다.

## 4. PR 전 체크리스트

- [ ] `python -m py_compile src/umic/**/*.py`로 문법 오류 없음 확인(CI가 자동으로도 돈다)
- [ ] `dry_run=True`로 구조 매칭 개수가 예상과 일치하는지 확인
- [ ] `umic.apply()`를 실제로 호출한 모델과 미수정 baseline을 비교해 정확도 확인
  (참고: [docs/260706_ve_production_integration.md](../docs/260706_ve_production_integration.md) §4의 검증 절차)
- [ ] `run_pipeline.py --mode both`로 성능 개선을 재확인
- [ ] 조사 과정을 `docs/YYMMDD_제목.md`로 남긴다([docs/REPORT_TEMPLATE.md](../docs/REPORT_TEMPLATE.md) 형식)
- [ ] README/CHANGELOG.md/`configs/expected_thor.yaml` 중 영향받는 부분을 갱신한다
  (README는 최신 수치만, 과거 이력은 CHANGELOG로)

## 5. 커밋/PR 메시지

`<type>: <one-line summary>` 형식을 쓴다(`feat`, `fix`, `docs`, `eval`, `chore` 등).
본문에는 무엇을·왜 바꿨는지와 측정 수치를 남긴다. 예시는 `git log`의 기존 커밋들을 참고한다.
