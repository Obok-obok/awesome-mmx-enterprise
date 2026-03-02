import math

from mmx.optimizer_core import ChannelConstraint, optimize_min_spend_for_target


def test_optimize_min_spend_reaches_target_within_cap():
    # Simple linear premium: P = 2*spend for A, 1*spend for B
    def pred_leads(ch, spend):
        return 0.0

    def pred_contracts(ch, spend):
        return 0.0

    def pred_premium(ch, spend):
        k = 2.0 if ch == "A" else 1.0
        return k * float(spend)

    channels = ["A", "B"]
    cur = {"A": 100.0, "B": 100.0}
    cons = {
        "A": ChannelConstraint(lock=False, min_ratio=0.0, max_ratio=10.0),
        "B": ChannelConstraint(lock=False, min_ratio=0.0, max_ratio=10.0),
    }

    # Need premium target 150 -> allocate 75 spend to A only (min spend)
    res = optimize_min_spend_for_target(
        channels=channels,
        current_spend=cur,
        constraints=cons,
        pred_leads=pred_leads,
        pred_contracts=pred_contracts,
        pred_premium=pred_premium,
        premium_target=150.0,
        step=5.0,
        budget_cap=500.0,
    )

    tot_spend = sum(res.recommended_spend.values())
    assert res.reached_target
    assert res.predicted_premium >= 150.0 - 1e-6
    # Greedy with step=5 should land at 75 exactly
    assert math.isclose(tot_spend, 75.0, rel_tol=0, abs_tol=1e-9)


def test_optimize_respects_locks_and_bounds():
    def pred_leads(ch, spend):
        return 0.0

    def pred_contracts(ch, spend):
        return 0.0

    def pred_premium(ch, spend):
        # make B much better, but lock B
        k = 10.0 if ch == "B" else 1.0
        return k * float(spend)

    channels = ["A", "B"]
    cur = {"A": 100.0, "B": 100.0}
    cons = {
        "A": ChannelConstraint(lock=False, min_ratio=0.5, max_ratio=1.0),
        "B": ChannelConstraint(lock=True, min_ratio=0.0, max_ratio=10.0),
    }

    res = optimize_min_spend_for_target(
        channels=channels,
        current_spend=cur,
        constraints=cons,
        pred_leads=pred_leads,
        pred_contracts=pred_contracts,
        pred_premium=pred_premium,
        premium_target=1200.0,
        step=10.0,
        budget_cap=1000.0,
    )

    # B must stay at current spend (lock)
    assert math.isclose(res.recommended_spend["B"], 100.0)
    # A cannot exceed max_ratio=1.0 -> <=100
    assert res.recommended_spend["A"] <= 100.0 + 1e-9
