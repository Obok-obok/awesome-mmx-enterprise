
## [5.4.2] - 2026-03-02

### Stability (Latent Quality Guardrails)
- (REFERENCE backend) Latent Quality(q) clip bounds를 train-window 관측 logit-residual 분위수(1%~99%)로 데이터 기반 자동 캘리브레이션.
- Downstream rate logit에 loading 계수 α 도입(기본 0.5) + q clip 적용으로 확률 폭주 방지.
- PPC×Quality 계수 gamma_ppc는 더 강하게 clamp/shrink(|gamma|≤0.1, 추가 shrink)하여 Premium 과대추정 리스크 완화.

### Optimization
- SLSQP objective(contracts×PPC)에서도 동일한 q clip + α 적용(최적화/예측 일관성).

### Dashboard
- Backtest 상단에 LQ clip/α/γ_ppc 디버그 배지(읽기 전용) 표시.
- Backtest artifacts config.json에 sem_globals 저장(감사/재현성).

## [5.4.1] - 2026-03-02

### Modeling
- (REFERENCE backend) Premium-per-Contract(PPC)에 Latent Quality(q) 연동 추가: log(PPC) = mu + gamma_ppc*q + eps.
- gamma_ppc는 train-window에서 q-hat(스팬드 기반 latent quality)로부터 강한 shrinkage 회귀로 캘리브레이션.

### Optimization
- 최적화 목적함수에 PPC×Quality 효과를 포함(옵션 A): q_bar(기간 평균)로 mu를 shift한 PPC 폐형식 모멘트를 사용.
- SLSQP 내부 objective는 결정론적으로 계산되어(샘플링 제거) λ 변경 시 안정적으로 반응.

### Dashboard
- Backtest KPI 표에서 'Premium / Contract'를 Amount 블록으로 이동(운영 관점 KPI 정렬), Unit Economics 중복 제거.

## [5.4.0] - 2026-03-02

### Modeling
- (REFERENCE backend) "Latent Quality" groundwork added: downstream funnel rates can vary over time via a latent quality state in logit-space.
- Latent quality is calibrated from train-window residuals and can optionally depend on spend intensity (log1p(spend)).

### Prediction Safety
- Funnel totals remain monotonic-clamped at aggregate level (attempts ≤ leads ≤ ...).

## [4.3.0] - 2026-03-01

### Changed
- Dashboard Home 구조 단순화 (p0 페이지 제거, app.py를 Home 역할로 정리)
- Executive / Recommendation 상단 해석 가이드 명확화
- Funnel 표 기본 컬럼을 예상/실제/차이 중심으로 단순화

### UX Standardization
- KPI 카드 단위 명시 (원/건/%)
- 표 상단에 예측 대상 기간 명시
- Risk-Adjusted 공식 및 P(AI>Human) 해석 가이드 보강

### Governance
- UI Spec 및 변경 이력 문서 정리

## [4.3.1] - 2026-03-01

### Fixed
- 대시보드 p3_recommendation.py 들여쓰기(IndentationError) 수정
- Home 페이지 정책 복구: p0_executive_overview를 Home으로 사용, app.py는 자동 리다이렉트

## [4.3.2] - 2026-03-01

### Dashboard Redesign
- Executive 화면에서 월별 라인차트(목표 vs 실적)를 "목표 대비 갭"(variance bar)로 교체
- "AI 추천 vs Do Nothing" 비교에서 Do Nothing 예산이 0으로 표시되는 문제를 방지(마트 기반 fallback)
- 추천/설명 영역의 raw JSON 출력 제거 → "결론-근거(3)-액션" 인사이트 카드로 표준화
- Recommendation 페이지(p3) 안정화 및 ViewModel 기반 추천 테이블(예산+근거 요약) 적용
