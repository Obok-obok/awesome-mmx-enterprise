# Architecture Overview

## Objective
Risk-Adjusted Premium = E[Premium] − λ · Std(Premium)

Premium = (Lead/Spend) * (Attempt/Lead) * (Connected/Attempt) * (Contracts/Connected) * (Premium/Contracts)

## Must-have Effects
- Adstock (Spend latent carryover) **필수**
- Saturation (Hill) **필수**
- Funnel-wide lag: Lead→Attempt, Connected→Contract **필수**
- Reporting delay: 기본 OFF, 품질 진단에서 체계적 지연 발견 시 옵션 ON

## Data Flow
Raw CSV → Mart(daily_channel_fact) → Bayesian SEM posterior → Optimization → Decision logs → Shadow eval → Monitoring → Feedback loop
