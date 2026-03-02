import pandas as pd


from mmx.tracking import (
    validate_actuals_df,
    compare_plan_vs_actual,
)


def test_validate_actuals_df_accepts_month_and_metrics():
    df = pd.DataFrame(
        {
            "month": ["2026-02"],
            "channel": ["A"],
            "spend": [1000],
            "leads": [10],
            "attempts": [7],
            "connected": [3],
            "contracts": [2],
            "premium": [5000],
        }
    )
    out = validate_actuals_df(df)
    assert out.loc[0, "month"] == "2026-02"
    assert float(out.loc[0, "spend"]) == 1000.0


def test_compare_plan_vs_actual_basic_math():
    plan = pd.DataFrame(
        {
            "month": ["2026-02", "2026-02"],
            "channel": ["A", "B"],
            "recommended_spend": [1000, 2000],
            "pred_leads": [10, 20],
            "pred_contracts": [2, 4],
            "pred_premium": [5000, 9000],
        }
    )
    actuals = pd.DataFrame(
        {
            "month": ["2026-02", "2026-02"],
            "channel": ["A", "B"],
            "spend": [1100, 1800],
            "leads": [11, 19],
            "attempts": [8, 14],
            "connected": [3, 5],
            "contracts": [3, 3],
            "premium": [5200, 8800],
        }
    )

    cmp_df, totals = compare_plan_vs_actual(plan, actuals, month="2026-02")

    # totals
    assert totals["recommended_spend"] == 3000.0
    assert totals["actual_spend"] == 2900.0

    # channel A variance
    a = cmp_df[cmp_df["channel"] == "A"].iloc[0]
    assert a["var_spend"] == 100.0
    assert a["var_premium"] == 200.0
