## 요약

<!-- 무엇을, 왜 바꿨는지 한두 문장. -->

## 측정

<!-- run_pipeline.py --mode both 결과, 또는 dry_run 매칭 개수 등. 숫자 없이 "빨라졌다"만 쓰지 않는다. -->

| 단계 | before | after | 개선 |
|---|---:|---:|---:|

## 정확도 검증

<!-- bit-exact인지, 어느 정도의 오차인지, 무엇과 비교했는지(untouched baseline vs umic.apply()). -->

## 체크리스트

- [ ] `python -m py_compile`로 문법 확인
- [ ] `dry_run=True`로 구조 매칭 개수 확인
- [ ] `umic.apply()` 실제 호출 vs 미수정 baseline으로 정확도 확인
- [ ] `run_pipeline.py --mode both`로 성능 재확인
- [ ] 조사 과정을 `docs/YYMMDD_제목.md`로 남김([docs/REPORT_TEMPLATE.md](../docs/REPORT_TEMPLATE.md) 형식)
- [ ] README/CHANGELOG.md/`configs/expected_thor.yaml` 중 영향받는 부분 갱신
