<<<<<<< HEAD
# MMX Bayesian Funnel SCM (Web→DB→TM→Contract→Premium)

생명보험사 **Web/DB → TM → 계약** 퍼널을 대상으로,
**Bayesian 구조방정식형(Structural Causal Model, SCM)** 을 구축해
- 채널 지출(구글/네이버/메타/토스/카카오)의 **DB(리드) 증분효과**
- DB 변화가 TM 연결/포화/응답지연을 통해 계약에 미치는 **간접효과**
- Posterior 샘플 기반 **시나리오(채널 OFF/예산 재배분) 시뮬레이션**
- **TM capacity 제약** 하의 예산 추천

을 데모 데이터로 end-to-end 재현합니다.

## 실행 (GCP VM-free)

=======
## Version
v3.2.1 (Design Master + Decision Markers)

# MMx Enterprise (Full) v3

CSV 기반 Enterprise MMx 시스템 풀 구현본입니다.

## 포함 기능 (요구사항 100%)
- Raw 이벤트 CSV → 일자×매체 Mart 생성 (원천 이벤트 기준, dedup, atomic write)
- Bayesian SEM (PyMC): Adstock(필수) + Saturation(필수) + Funnel-wide lag(Lead→Attempt, Connected→Contract)
- Reporting delay: 기본 OFF, 진단 신호 기반 옵션 활성화 가능
- Optimization: SLSQP + Risk-Adjusted Premium (기본) + δ 안정성 제약
- Explainability: 반응곡선, 한계 ROI(mROI), 포화도/EC50, half-life
- Shadow Mode 평가: AI vs Human, Counterfactual(Posterior Predictive) 기반 ΔRA, P(AI>Human)
- Governance: 모델 레지스트리/승격 로그, Rollout policy, 실험 레지스트리
- Streamlit Dashboard: 기준시점/커버리지 표시, 표 합계, Business label 적용, 스토리라인 페이지
- FastAPI: 이벤트 인입 + 파이프라인 + 추천 + 평가 + 모니터링
- Docs PDF 생성 스크립트 포함

## 설치
>>>>>>> eebb871 (version up)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

<<<<<<< HEAD
bash scripts/run_all.sh
streamlit run app_mmx_dashboard.py --server.port 8502 --server.address 0.0.0.0
```

## (NEW) Adaptive 모드: Dynamic Bayesian SCM + Online 업데이트

기존 배치(PyMC MCMC/ADVI) 학습 외에, **자동 적응형(Adaptive) 운영**을 위해 아래 기능을 추가함.

- **Dynamic Bayesian SCM**: 채널 미디어 효과(탄력)가 시간에 따라 변할 수 있도록 **Kalman Filter 기반 DLM**으로 추정
- **Online Bayesian Updating**: 연결률/성사율(베타-이항), 계약당 보험료(정규-역감마)를 **일 단위로 순차 업데이트**
- **Next-day 예산 추천**: 사후분포 샘플링(Thompson-style)로 **다음날 예산 배분**을 bounds 하에서 추천

실행:

```bash
python -m src.pipeline.run_dynamic_online
```

산출물(`out/dynamic/`):

- `online_fit_log.csv` : 일 단위 온라인 업데이트 로그
- `dlm_state_means.parquet` : time-varying state(채널 효과 포함) 필터링 평균
- `dlm_state_means.csv` : (parquet 엔진 없이도 실행되도록 CSV로 제공)
- `budget_reco_next_day.csv` : 다음날 예산 추천

## (NEW) 운영 모드: Daily 자동 루프 + Drift 감지 + Risk-aware 예산

운영 단계에서는 **매일 1회** 아래 루프를 자동 실행하는 구성이 핵심임.

1) 전날 데이터로 **온라인 업데이트(1일분)**
2) Leads residual 기반 **Drift 감지** → 심각하면 DLM 적응속도(Q) 상향
3) 다음날 예산: **Thompson + Exploration(eps)** + **CVaR(하방 10%) 점수** 기반 추천
4) **실제 성과(보험료) vs 사후예측**을 `out/ops/performance_daily.csv`에 누적

수동 실행(테스트):

```bash
bash scripts/run_ops_daily.sh
```

산출물:

- `out/dynamic/adaptive_state.json` : 모델 상태(재시작/재부팅에도 이어서 학습)
- `out/dynamic/budget_reco_next_day.csv` : 다음날 채널별 예산 추천
- `out/ops/performance_daily.csv` : 실제 보험료 vs 예측(Mean/CVaR10) + drift 로그

### systemd timer로 매일 자동 실행 (권장: GCP VM)

1) 서비스/타이머 설치(경로는 홈 기준 예시)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/mmx-ops.service ~/.config/systemd/user/
cp deploy/systemd/mmx-ops.timer  ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now mmx-ops.timer
systemctl --user status mmx-ops.timer
```

2) 로그 확인

```bash
journalctl --user -u mmx-ops.service -n 200 --no-pager
```

> `mmx-ops.service`의 WorkingDirectory 는 `%h/workspace/mmx_bayes_campaign_adaptive_dynamic` 로 되어 있음.
> 본인 폴더 경로에 맞게 수정하면 됨.

### 대시보드에서 성과 비교 보기

Streamlit 대시보드의 **Ops Monitoring** 탭에서
- 실제 보험료 vs 사후예측(Mean / CVaR10)
- Drift z-score
- 다음날 예산 추천
을 한 화면에서 확인 가능.

---

## (NEW) Plan vs Actual (월 단위 Tracking)

대시보드에서 "예산을 배분했고(Plan)", "성과를 예측했는데(Forecast)", "실제로는 성과가 어떻게 나왔는지(Actual)"를 **정밀 정합성 체크**와 함께 월 단위로 비교할 수 있도록 탭을 추가함.

### 사용 흐름

1) **4) Budget Optimizer (Act)** 탭
- 최적화 실행 → 하단의 **✅ 플랜 저장** 클릭
- `out/runs/<run_id>/plan_by_channel.csv` 및 `plan_summary.json` 생성

2) **7) Plan vs Actual (Track)** 탭
- Run 선택 + 비교 월(YYYY-MM)
- `actuals.csv` 업로드(또는 샘플 actuals)
- 채널별/합계 Plan vs Actual + 차이(variance) + ROI 비교

### Actuals CSV 스키마

- `month`(YYYY-MM) 또는 `date`(YYYY-MM-DD)
- `channel`, `spend`, `leads`, `contracts`, `premium`

### 테스트

```bash
pip install -r requirements.txt
pytest -q
```

## 입력 데이터(심플 레이아웃: 2 파일)

운영 적용을 위해 **채널 패널(안정적 추정)** + **캠페인 패널(메시지/소재 레벨 분해)** 2개 파일만 받습니다.

### 1) `data/input_daily_channel.csv` (date × channel)

| column | 설명 |
|---|---|
| date | 일자 |
| channel | google/naver/meta/toss/kakao |
| spend | 일 지출 |
| leads | 일 DB 건수 |
| tm_attempts | 콜 시도 |
| tm_connected | 연결 |
| contracts | 계약(증권건수) |
| premium | 월 보험료 합 |

### 2) `data/input_daily_campaign.csv` (date × channel × campaign)

| column | 설명 |
|---|---|
| date | 일자 |
| channel | google/naver/meta/toss/kakao |
| campaign_id | 캠페인 ID |
| message_type | SA/DA 등 메시지 타입 |

---

## (NEW) A/B Holdout 통합 (탐색 예산을 실험군으로 고정)

관측 데이터 기반 인과추정의 약점을 줄이기 위해, **탐색(exploration) 일부를 주간 단위로 CONTROL/TREATMENT로 고정**하는 간단한 홀드아웃 실험을 추가할 수 있음.

### 설정

`configs/mmx.yaml`:

- `ab_holdout.enabled: true`
- `ab_holdout.holdout_channels: [google, naver]` 처럼 실험 대상 채널 지정
- `ab_holdout.delta: 0.20` (TREATMENT +20%, CONTROL -20%)

### 산출물

- `out/ops/ab_plan_next_day.csv` : 내일 실험 계획(채널/그룹/승수/예산)
- `out/ops/ab_assignments.csv` : 날짜별 실험 할당 히스토리
- `out/ops/ab_results_weekly.csv` : 최근 7일의 단순 diff-of-means 기반 주간 결과(Starter)

> 운영 고도화 시에는 Geo split / matched market / uplift 모델 등을 붙여 식별을 더 강화할 수 있음.

---

## (NEW) Geo Holdout / Matched Market (진짜 식별 강화)

채널 A/B 홀드아웃보다 더 강한 식별을 위해, **지역(geo) 단위 홀드아웃 + 매칭(matched market)** 을 사용할 수 있음.

핵심 로직:
- 과거 `premium` 트렌드 상관이 높은 geo를 **쌍(pair)** 으로 매칭
- 매주(ISO week) 단위로 각 pair에서 CONTROL/TREATMENT를 **고정 할당**
- `holdout_channels`에 한해 geo별로 **+/-delta** 예산을 적용 (채널 총합은 유지되도록 재정규화)
- 주간 결과는 matched market 간단 DiD로 lift를 추정

### 입력(옵션)

`data/input_daily_geo_channel.csv`:

| column | 설명 |
|---|---|
| date | 일자 |
| geo | 지역 코드/명 |
| channel | google/naver/meta/toss/kakao |
| spend/leads/tm_attempts/tm_connected/contracts/premium | 채널 파일과 동일 의미 |

### 설정

`configs/mmx.yaml`에서 `geo_holdout.enabled: true` 후 아래 조정:
- `geo_holdout.holdout_channels`
- `geo_holdout.delta`

### 산출물

- `out/ops/geo_plan_next_day.csv` : 내일 geo×channel 예산 계획
- `out/ops/geo_assignments.csv` : geo 할당 히스토리
- `out/ops/geo_mm_results_weekly.csv` : matched market 주간 lift

---

## (NEW) CVaR 기반 예산 최적화: 하방 손실 제한(하드 제약)

운영에서 CFO/리스크 관점으로는 평균(Mean)보다 **하방(CVaR10)** 이 중요함.

`configs/mmx.yaml` → `ops_loop`:
- `cvar_floor_ratio`: equal-split baseline CVaR10 대비 최소 비율(예: 0.95)
- `cvar_floor_abs`: 절대 최소 CVaR10(옵션)

동작:
- 후보 예산안을 여러 개 샘플링 → **예측 CVaR10이 floor 미만이면 폐기**
- 제약을 만족하는 후보 중 예측 Mean이 최대인 안을 채택

---

## (NEW) 주간 리포트 자동 PDF + 이메일 발송

### 실행

```bash
bash scripts/run_weekly_report.sh
```

생성된 PDF:

- `out/reports/weekly_report_<YYYY-MM-DD>.pdf`

> 리포트는 **1페이지 고정(임원 보고 규격)** 템플릿으로 생성됨. (섹션/표 구성 고정)

### 이메일 발송 활성화

`configs/mmx.yaml`에서:

- `reporting.email.enabled: true`

SMTP 환경변수 설정(예시):

```bash
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USER="user@example.com"
export SMTP_PASS="..."
export SMTP_FROM="user@example.com"
export SMTP_TO="boss@example.com,team@example.com"
export SMTP_TLS="true"
```

### systemd (주간 자동 실행)

`deploy/systemd/mmx-weekly.service`, `deploy/systemd/mmx-weekly.timer` 템플릿 제공.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/mmx-weekly.service ~/.config/systemd/user/
cp deploy/systemd/mmx-weekly.timer  ~/.config/systemd/user/

# WorkingDirectory를 본인 경로로 1회 수정 후
systemctl --user daemon-reload
systemctl --user enable --now mmx-weekly.timer
systemctl --user status mmx-weekly.timer
```
| spend | 일 지출 |
| leads | 일 DB 건수 |
| tm_attempts | 콜 시도 |
| tm_connected | 연결 |
| contracts | 계약(증권건수) |
| premium | 월 보험료 합 |

> 실제로 캠페인 단위 계약 매핑은 보통 `customer_id`/`lead_id`로 연결함.
> 본 데모는 (선택) `data/raw_events.csv`를 함께 생성해 “고객번호 기반 캠페인→계약 연결” 흐름을 보여줌.

## 산출물

- `out/executive_summary.json`
- `out/posterior_summary.csv`
- `out/counterfactuals.csv`
- `out/budget_recommendation.csv`
- `out/metric_lineage.csv`

## customer_id 기반 캠페인→증권 매핑 규칙 (운영 기본)

원천 테이블을 보유한 경우(리드/콜/증권) `customer_id`로 캠페인 패널을 생성할 수 있음.

기본 규칙(ops-safe baseline):

1) **30일 윈도우**: 계약(증권)은 계약시각 기준 30일 이내의 리드만 후보임

2) **중복 리드 처리**: 동일 고객이 동일 (channel, campaign, message)로 **24시간 이내** 여러 번 리드 생성 시, **최초 1건만 유지**

3) **다중 계약 처리**: 동일 고객이 여러 증권을 보유한 경우, **각 증권을 개별적으로** 계약시각 이전의 가장 최근 리드(Last-touch)로 매핑함

### 입력(원천) 파일 예시

- `raw_leads.csv`: `customer_id, lead_ts, channel, campaign_id, message_type`
- `raw_tm_calls.csv`: `call_id, customer_id, call_ts, connected_flag` (+ 가능하면 channel/campaign/message)
- `raw_policies.csv`: `policy_id, customer_id, contract_ts, premium`
- (선택) `spend_campaign.csv`: `date, channel, campaign_id, message_type, spend`

### 원천→모델 입력 2파일 생성

```bash
python -m src.cli build-inputs \
  --raw-dir data/sample \
  --spend-campaign data/sample/spend_campaign.csv \
  --out data/inputs \
  --window-days 30 \
  --dedupe-hours 24
```

생성 결과:
- `data/inputs/input_daily_channel.csv`
- `data/inputs/input_daily_campaign.csv`
- `data/inputs/data_quality_report.json`


## Performance / Low-memory VM tips
- Set `ops_loop.max_history_days` (default 240) to limit history kept in memory.
- For geo matched market, set `geo_holdout.mm_max_geos` (default 40) to cap geos used for matching.
- If the VM still OOMs, add swap (e.g., 2GB) and/or run weekly jobs off-peak.
=======
# 로컬 모듈 import (DB/패키징 없이 실행)
export PYTHONPATH="$(pwd)/src"
```

## 실행 (권장 순서)
```bash
python scripts/make_docs_pdf.py
python scripts/build_mart.py
python scripts/train_model.py
python scripts/promote_model.py --model-version <TRAIN_OUTPUT>
python scripts/recommend.py --period-start 2026-04-01 --period-end 2026-04-30 --total-budget 100000000
streamlit run apps/dashboard/app.py
```

## API (선택)
```bash
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

## One-Click Demo (v3.1)
```bash
python scripts/generate_sample_data.py
bash scripts/demo_run_all.sh
```

## Backtest (Synthetic, Engine vs Known Outcomes)

1) 모조 Raw Event 생성 + Mart/Model/Backtest까지 원커맨드:
```bash
python scripts/run_backtest.py --mode demo --seed 42 --start 2025-01-01 --end 2026-03-31 --test-days 90 --n-samples 1200
```

2) 대시보드에서 확인:
```bash
streamlit run apps/dashboard/app.py
```

Backtest 결과는 `artifacts/backtests/latest/`에 저장되며, UI는 해당 아티팩트만 읽어 렌더링합니다(임의 숫자 생성 없음).


## Program Inventory
See `docs/src/program_inventory.md` for the enterprise inventory (purpose / inputs / outputs / dependencies / artifacts).


> Note: The dashboard includes an import path guard so `streamlit run apps/dashboard/app.py` works even when the repo root isn't on PYTHONPATH.


## Integrity Validation
Run:
```bash
python scripts/validate_integrity.py
```
>>>>>>> eebb871 (version up)
