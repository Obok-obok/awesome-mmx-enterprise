import pandas as pd
import streamlit as st

def render_metrics_dictionary():
    st.title("지표 정의 · 산식 · 해석 가이드 (Metrics Dictionary)")
    st.markdown(
        """이 페이지는 대시보드에서 사용되는 **핵심 지표의 정의/산식/해석 방법/주의사항**을 한 곳에 정리한 운영 문서임.

- 목적: 해석 일관성 확보, 지표 변경 영향 추적, 운영/감사 대응
- 원천: `out/panel_daily_channel.csv` (기본), 일부는 Optimizer/시뮬레이션 산출물(`st.session_state`)을 사용
"""
    )

    def show(rows, title):
        st.subheader(title)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    show([
        {"지표":"이번달 목표 보험료(Target Premium)","정의":"해당 월(또는 선택 기간) 목표 Premium","산식":"사용자 입력(콤마 자동 마스크)","해석 방법":"목표 달성률/갭 계산의 기준선","주의사항":"기간 모드가 Selected Range일 경우 ‘선택기간 목표’로 재해석 필요"},
        {"지표":"현재 MTD 보험료(MTD Premium)","정의":"분석 기간 내 Premium 합","산식":"Σ premium","해석 방법":"성과의 최상위 결과 지표","주의사항":"Premium 정의(수입보험료/계약보험료 등) 조직 표준 확인"},
        {"지표":"목표 달성률(Achievement)","정의":"목표 대비 현재 달성 비율","산식":"MTD Premium / Target Premium","해석 방법":"80% 이상이면 잔여 기간 전략은 ‘유지/효율’ 중심","주의사항":"Target 입력 단위 오류 시 해석 왜곡(만원/원 혼동)"},
        {"지표":"잔여 필요 보험료(Goal Gap)","정의":"목표까지 남은 Premium","산식":"Target - MTD","해석 방법":"추가 액션 규모를 결정하는 1차 입력","주의사항":"음수(초과달성)면 ‘추가 집행 필요 없음’으로 해석"},
        {"지표":"일평균 필요 보험료","정의":"월말(또는 분석 종료일 기준)까지 일평균 필요한 Premium","산식":"Gap / Remaining Days","해석 방법":"운영 페이스(속도) 관리 지표","주의사항":"Remaining Days=0이면 ‘-’로 표기"},
    ], "1) Executive KPI (Decide)")

    show([
        {"지표":"예산(Spend)","정의":"광고비 합","산식":"Σ spend","해석 방법":"투입(레버) 지표","주의사항":"매체/캠페인 기준 조인 누락 시 과소계상"},
        {"지표":"Leads","정의":"리드 수 합","산식":"Σ leads","해석 방법":"중간성과(퍼널 상단)","주의사항":"중복 리드/가짜 리드 정제 필요"},
        {"지표":"Sales(Contracts)","정의":"계약/판매 건수 합","산식":"Σ contracts","해석 방법":"퍼널 하단 결과(매출 전 단계)","주의사항":"계약 인정 기준(유효/취소) 확인"},
        {"지표":"RR(Conversion Rate)","정의":"Lead→Sales 전환율","산식":"Contracts / Leads","해석 방법":"세일즈/심사/콜 품질의 대표 지표","주의사항":"Leads가 작을 때 변동성 급증"},
        {"지표":"건당 보험료(Premium per Sale)","정의":"계약 1건당 Premium","산식":"Premium / Contracts","해석 방법":"상품 믹스/단가 변화를 반영","주의사항":"Contracts=0이면 ‘-’ 또는 0으로 표기"},
        {"지표":"총보험료(Total Premium)","정의":"Premium 합","산식":"Σ premium","해석 방법":"최종 결과 지표","주의사항":"세그먼트/채널 집계 기준 일관성 필요"},
    ], "2) Funnel KPI")

    show([
        {"지표":"ROI","정의":"Spend 대비 Premium","산식":"Premium / Spend","해석 방법":"예산 재배분 우선순위(높을수록 좋음)","주의사항":"증분효과가 아닌 ‘관측 ROI’임"},

        {"지표":"Leads/Spend","정의":"Spend 대비 Leads","산식":"Leads / Spend","해석 방법":"지출 1원당 리드 획득량(상단 퍼널)","주의사항":"Spend=0이면 NaN, 리드 정의(유효리드) 표준 필요"},
        {"지표":"Attempts/Leads","정의":"Leads 대비 통화시도 횟수","산식":"TM Attempts / Leads","해석 방법":"리드 1건당 재시도 포함 처리 강도(운영 레버)","주의사항":"리드 분모=0이면 NaN, 자동/수동 재시도 정책에 민감"},
        {"지표":"Connected/Attempts","정의":"통화시도 대비 연결율","산식":"TM Connected / TM Attempts","해석 방법":"콜센터 포화/리스트 품질 신호","주의사항":"시도 분모=0이면 NaN"},
        {"지표":"Contracts/Connected","정의":"연결 대비 계약 전환율","산식":"Contracts / TM Connected","해석 방법":"상담 품질/상품 경쟁력/리드 질","주의사항":"연결 분모=0이면 NaN"},
        {"지표":"Premium/Contract","정의":"계약당 평균 보험료","산식":"Premium / Contracts","해석 방법":"단가/상품 믹스(APC)","주의사항":"계약 분모=0이면 NaN"},
        {"지표":"5-Factor 곱(재구성 ROI)","정의":"5개 factor 곱으로 복원한 ROI","산식":"(Leads/Spend)*(Attempts/Leads)*(Connected/Attempts)*(Contracts/Connected)*(Premium/Contract)","해석 방법":"실제 ROI와 차이가 0에 가까울수록 정의/집계 일치","주의사항":"기간 필터/분모 0 처리 규칙이 다르면 차이가 발생"},

        {"지표":"CPL","정의":"Lead 1건당 비용","산식":"Spend / Leads","해석 방법":"상단 퍼널 효율","주의사항":"Leads 품질(유효 리드)와 함께 봐야 함"},
        {"지표":"CPS","정의":"Sales 1건당 비용","산식":"Spend / Contracts","해석 방법":"하단 퍼널 효율","주의사항":"Contracts 지연/취소 반영 여부 확인"},
        {"지표":"RR","정의":"Lead→Sales 전환율","산식":"Contracts / Leads","해석 방법":"세일즈/운영 영향이 큰 지표","주의사항":"리드 믹스 변하면 RR도 변함"},
        {"지표":"건당 보험료","정의":"Sales 1건당 Premium","산식":"Premium / Contracts","해석 방법":"단가/상품 믹스 영향","주의사항":"프로모션/할인 시 급변 가능"},
    ], "3) Breakdown (채널별)")

    st.subheader("4) Budget Optimizer (Act) — 엔진 가정/로직")
    st.markdown(
        r"""Optimizer는 실무 제약(채널 Lock/Min/Max, 총예산 고정) 하에서 예산을 재배분하는 **휴리스틱 최적화 엔진**임.

- Spend→Leads: 최근 윈도우(기본 8주) 데이터로 포화(체감) 곡선을 근사(로그 기반)하거나 신호가 약하면 선형 효율로 fallback  
- Leads→Contracts: 채널별 RR(선택기간 또는 캘린더월) 적용  
- Contracts→Premium: 채널별 건당보험료(선택기간 또는 캘린더월) 적용  
- **(기본) 목적함수: Premium Target 달성 비용 최소화**  
  - 결정변수: 채널별 예산 \(x_c\)  
  - 예측 Premium: \(\hat{P}(x)=\sum_c \hat{P}_c(x_c)\) (Spend→Leads 포화 + RR + APS 체인 기반)  
  - 제약: \(x_c \in [\alpha_c\,x^{cur}_c,\;\beta_c\,x^{cur}_c]\) (Min/Max), Lock이면 \(x_c=x^{cur}_c\), 그리고 \(\sum_c x_c \le B\) (예산 상한)  
  - 목표: \(\min_{x}\; \sum_c x_c\) s.t. \(\hat{P}(x)\ge P^{target}\)  
  - 해석: **목표 보험료를 달성하는데 필요한 최소 집행액**을 우선 산출하고, 남는 예산은 ‘추가 uplift’ 용도로 분리함

- (보조) 목적함수: 예산 고정 하 Premium 최대화  
  - \(\max_{x}\; \hat{P}(x)\) s.t. \(\sum_c x_c=B\) 및 동일 제약(락/Min/Max)
- 출력: 채널별 현재/권장 예산, Δ 예산, 예상 Leads/Contracts/Premium 변화(가정 기반)
"""
    )

    st.subheader("6) Engine Value(엔진 가치) — 수식")
    st.markdown(
        r"""엔진 가치는 ‘동일 조건에서, 엔진이 권장한 예산 배분이 **Legacy 운영 대비** 얼마나 더 나은 성과를 내는지’로 정의함.

- Legacy baseline(운영 방식) 정의(기본): History 기간의 **채널 예산 비중(share)** 을 그대로 유지
  - \(s^{legacy}_c = \dfrac{Spend^{hist}_c}{\sum_j Spend^{hist}_j}\)
  - 비교 예산(동일 예산 가정): \(x^{legacy}_c = s^{legacy}_c \cdot \sum_j x^{engine}_j\)

- Forecast 기반 엔진 가치(예측 uplift)
  - \(Value^{forecast} = \hat{P}(x^{engine}) - \hat{P}(x^{legacy})\)
  - ROI uplift: \(\dfrac{\hat{P}(x^{engine})}{\sum x^{engine}} - \dfrac{\hat{P}(x^{legacy})}{\sum x^{legacy}}\)

- Actual 기반 엔진 가치(실적 uplift)
  - (가능하면) Impact 기간 Actual을 사용해 \(Value^{actual} = P^{actual}_{impact} - P^{legacy,actual}_{impact}\)
  - Legacy actual은 (초기 버전) 실제 집행이 Legacy로 실행되지 않았으므로 **대체 기준선**(예: 과거 동일 월/동일 share 유지 시나리오)로 보고, 실험(holdout) 설계로 고도화 권장
"""
    )

    st.subheader("5) 목표 달성 역산 (Channel-level Required Spend Decomposition)")
    st.markdown(
        """목표 미달(Gap)이 있을 때, **추가 예산을 어느 채널에 집행하면 목표 달성(커버리지)이 증가하는지** 추정하는 섹션임.

- 채널 한계효과: (Spend+step)에서의 예측 Premium 증가(ΔPremium/step)  
- 배분 로직: ΔPremium/step이 큰 채널부터 step 단위로 그리디 배분(포화/상한 고려)  
- 출력: 추가 예산 배분표, 예상 Premium 증가, 목표 커버리지(%)  
- 해석: 커버리지는 ‘확률’이 아니라, 현재 관측 효율/가정 기반의 **추정치**임 (실험/증분효과로 보정 권장)
"""
    )

    st.info("지표 정의/산식은 데이터 구조(out/*)와 가정에 따라 확장 가능. 지표 변경 시 본 페이지를 ‘단일 진실원(Single Source of Truth)’으로 업데이트 권장.")
