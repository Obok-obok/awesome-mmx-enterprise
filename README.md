
# MMX Enterprise – Marketing Media Mix AI Agent

**문서 생성일:** 2026-03-02

---

# 1. 시스템 목적 (비즈니스 관점)

MMX Enterprise는 보험 TM(텔레마케팅) 환경에서 디지털 광고 예산을
데이터 기반으로 최적 배분하여 **Risk-Adjusted Premium을 극대화**하는
엔터프라이즈 의사결정 AI 플랫폼입니다.

기존 운영 방식은 다음과 같은 한계를 가집니다:

- 리드 규모 기반의 경험적 예산 배분
- 포화(Saturation) 채널 과투자
- 수익 변동성(Risk) 미반영
- 퍼널 단계별 원인 분석 어려움

본 시스템은 확률 기반 구조방정식 모델과 제약 최적화를 통해
매출 극대화와 리스크 통제를 동시에 달성합니다.

---

# 2. 시스템 전체 아키텍처

시스템은 4개 주요 레이어로 구성됩니다.

## 2.1 Data Layer

### Raw Event (Append-only)
- event_id
- occurred_at
- channel
- spend_amount
- lead_count
- call_attempt_count
- call_connected_count
- contract_count
- premium_amount

### Mart Table (daily_channel_fact)
단위: date × channel

컬럼:
- spend
- leads
- attempts
- connected
- contracts
- premium

정합성 검증:
- attempts ≤ leads
- connected ≤ attempts
- contracts ≤ connected
- 모든 전환율 ∈ [0,1]

---

## 2.2 Modeling Engine (Bayesian Structural Equation Model)

### 퍼널 구조

Spend → Leads → Attempts → Connected → Contracts → Premium

### 단계별 모델

(1) Spend → Leads  
- Adstock (Carryover 효과 반영)  
- Hill Saturation (수확 체감 반영)

(2) Funnel 전환  
- Binomial 기반 확률 모델

(3) Contracts → Premium  
- LogNormal 기반 계약당 프리미엄 모델

---

## 2.3 Optimization Engine

목적함수:

RA Premium = E[Premium] − λ · Std(Premium)

제약조건:
- Σ spend_i = total_budget
- 채널별 min/max (선택)
- 안정성 제약
- 정책 기반 rollout 제약

최적화는 폐형식 모멘트 기반 Deterministic 방식 사용.

---

## 2.4 Dashboard & Backtesting

- Historical Backtest
- Baseline vs Model Allocation 비교
- Funnel 단계별 Actual vs Predicted 비교
- Risk-Adjusted 성과 비교

---

# 3. 적용 이론

본 시스템은 다음 이론 기반:

- Bayesian Structural Equation Modeling
- Hill Saturation Function
- Adstock Carryover Model
- LogNormal Revenue Modeling
- Risk-Adjusted Portfolio Optimization
- Constrained Nonlinear Optimization (SLSQP)

---

# 4. 계산 원리

Premium 계산 구조:

Premium =
(Lead / Spend) ×
(Call Attempt / Lead) ×
(Call Connected / Call Attempt) ×
(Contracts / Call Connected) ×
(Premium / Contracts) × Spend

또는

Premium = Contracts × Premium per Contract

Risk-Adjusted Premium:

RA = E[Premium] − λ · Std(Premium)

Posterior 평균 또는 모멘트 근사 기반으로 계산됩니다.

---

# 5. 모델 엔진 계산 흐름 (로직 흐름)

1. 데이터 로드 (daily_channel_fact)
2. 채널별 시계열 정렬
3. Spend → Leads 예측 (Adstock + Hill)
4. Leads → Attempts 예측 (Binomial mean path)
5. Attempts → Connected 예측
6. Connected → Contracts 예측
7. Contracts × PPC → Premium 계산
8. Premium 분산 추정
9. RA Premium 계산
10. SLSQP 최적화 수행

---

# 6. 프로그램 실행 흐름

## Backtest 실행 흐름

1. CLI 입력 파싱
2. 데이터 생성 또는 로드
3. Train/Test 분리
4. Posterior 추정
5. Baseline 성과 계산
6. Optimization 수행
7. 결과 저장 (artifacts)
8. Dashboard에서 시각화

---

# 7. 폴더 구조 (Repository Overview)

awesome_mmx/
│
├── apps/
│   └── dashboard/          # Streamlit UI
│
├── configs/
│   └── sim/                # 시뮬레이션 설정
│
├── scripts/
│   └── run_backtest.py     # 실행 엔트리포인트
│
├── src/mmx/
│   ├── engine/
│   │   ├── sem/            # 구조방정식 모델
│   │   ├── optimizer/      # 최적화 로직
│   │   └── simulation/     # 데이터 생성기
│   │
│   ├── usecases/
│   │   └── run_backtest.py
│   │
│   └── domain/
│
└── artifacts/              # 실행 결과 저장

---

# 8. 실행 방법

## Backtest 실행

```bash
python scripts/run_backtest.py   --mode demo   --seed 42   --start 2025-01-01 --end 2026-03-31   --test-days 90   --backend REFERENCE   --n-samples 200
```

## Dashboard 실행

```bash
streamlit run apps/dashboard/app.py
```

URL : https://awesome-fds-9qyj9byrubzicscijfsgnh.streamlit.app/ 

---

# 9. 시스템 특징 요약

- Risk-aware Premium 최적화
- 포화 기반 예산 재배분
- 확률 기반 퍼널 예측
- Deterministic 최적화 엔진
- 엔터프라이즈 운영 대응 구조

---
