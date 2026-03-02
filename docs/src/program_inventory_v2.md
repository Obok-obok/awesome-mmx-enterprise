
# MMX Enterprise Program Inventory v2.0
Generated: 2026-03-01T00:16:57.757932 UTC

---

# 1. SYSTEM ENTRYPOINTS

## scripts/build_mart.py
### Purpose
Aggregate raw event-level marketing data into daily channel mart with strict validation gates.

### Key Responsibilities
- Validate schema and required columns
- Enforce Data Quality Gate:
  - Timestamp NaT ratio threshold
  - Channel allowlist enforcement
  - Unknown channel ratio threshold
  - Funnel monotonicity validation
- Generate daily_channel_fact.csv
- Generate mart_validation_latest.json

### Input
- data/raw/events/* CSV files

### Output
- data/curated/mart/daily_channel_fact.csv
- logs/pipeline/mart_validation_latest.json

### Upstream
- Raw event ingestion

### Downstream
- train_model.py
- Data Quality Gate (Dashboard p5)

### Related Requirements
- Event-based input
- Data Quality Governance
- Funnel monotonicity enforcement

---

## scripts/train_model.py
### Purpose
Train Bayesian SEM model.

### Responsibilities
- Fit Adstock + Saturation + Lag model
- Persist posterior samples
- Store model version metadata

### Input
- daily_channel_fact.csv

### Output
- artifacts/models/<version>/posterior.pkl

### Downstream
- promote_model.py
- recommend.py

---

## scripts/promote_model.py
### Purpose
Promote model version to production.

### Output
- artifacts/models/promoted.txt

---

## scripts/recommend.py
### Purpose
Run budget optimization and produce decision artifact.

### Responsibilities
- Synchronize n_days with selected period
- Apply warm-start carryover
- Compute Risk-Adjusted Premium
- Enforce:
  - Budget equality
  - Bounds
  - δ stability constraint
  - Ramp-up cap
- Generate validation_report
- Compute data_coverage
- Produce Funnel Forecast artifact

### Output
- artifacts/recommendations/<decision_id>.json
- artifacts/recommendations/funnel_forecast/<decision_id>.csv

---

# 2. ENGINE LAYER (src/mmx/engine)

## adstock.py
Implements geometric carryover effect.

## saturation.py
Implements Hill response curve and EC50 saturation ratio.

## pymc_sem.py
Defines Bayesian SEM:
- Spend → Lead (Adstock + Saturation)
- Lead → Attempt (Lag)
- Attempt → Connected
- Connected → Contract (Lag)
- Premium = Contracts × PPC

## inference.py
Posterior predictive simulation:
- Warm-start window
- Period synchronization
- Shared sampling for decision & reporting consistency

---

# 3. OPTIMIZATION LAYER (src/mmx/optimization)

## objective.py
Risk-Adjusted Premium:
RA = E[Premium] − λ·Std[Premium]

## solver.py
SLSQP solver with equality constraint.

## constraints.py
- Bounds
- δ stability
- Ramp-up cap
- Post-solution validation_report

---

# 4. DATA QUALITY & GOVERNANCE

## data_quality.py
- Weekend spike diagnostic
- Reporting delay suggestion

## Mart Validation Log
logs/pipeline/mart_validation_latest.json

Exposed in Dashboard Governance (p5).

---

# 5. DASHBOARD LAYER (apps/dashboard)

## bootstrap.py
Global initialization:
- Filters
- Context bar
- Styling
- Path safety

## pages
- p0: Executive Overview
- p1: Funnel Diagnostics
- p2: Channel Response Lab
- p3: Recommendation
- p4: Shadow Evaluation
- p5: Governance & Data Quality

---

# 6. ARTIFACT TRACEABILITY

## Decision Artifact
Contains:
- Expected Premium
- Premium Std
- CI bounds
- validation_report
- data_coverage
- warm_start_days

## Funnel Forecast Artifact
Stage-level forecast consistency.

---

# 7. REQUIREMENT TRACEABILITY LINK

See:
docs/src/requirements_traceability.md

---

# 8. VERSION HISTORY

## v2.0
- Full traceable architecture mapping
- Governance & Data Quality fully integrated
- Optimization validation integrated
- Warm-start synchronization enforced
