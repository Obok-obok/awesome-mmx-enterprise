"""Adaptive / Dynamic MMX modules.

This package adds:
1) Dynamic Bayesian SCM (time-varying channel effects) via a lightweight
   Dynamic Linear Model (DLM) with Kalman filtering.
2) Online Bayesian updating for funnel rates (beta-binomial) and premium
   (normal-inverse-gamma).
3) Budget allocation using posterior sampling (Thompson-style) over
   short-horizon counterfactual simulations.

Design goal: run on a small GCP VM-free environment with minimal deps
(numpy/pandas/scipy) and deterministic outputs.
"""
