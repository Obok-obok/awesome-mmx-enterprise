# Metric Dictionary (Reproducible)

## Mart
`data/curated/mart/daily_channel_fact.csv`

Columns:
- date, channel
- spend, leads, call_attempt, call_connected, contracts, premium

## Derived rates
- lead_per_spend = leads/spend
- attempt_per_lead = call_attempt/leads
- connected_rate = call_connected/call_attempt
- contract_rate = contracts/call_connected
- premium_per_contract = premium/contracts

## Explainability
- Response curve
- mROI curve
- EC50, Saturation ratio
- Half-life
