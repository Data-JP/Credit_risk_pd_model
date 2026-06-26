# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:47:07 2026

@author: J-P-F
"""

"""Evaluation utilities: decile gains table."""
import pandas as pd

def gains_table(y_true, y_pred):
    """
    Build a decile-level gains table from true binary labels and predicted probabilities.

    Parameters
    ----------
    y_true : pandas.Series
        Binary ground-truth labels (0/1). Only the values are used.
    y_pred : array-like or pandas.Series
        Predicted probabilities or scores (higher means higher predicted default risk).

    Returns
    -------
    pandas.DataFrame
        Gains table indexed by `decile` (1 = riskiest / highest PD ... 10 = safest)
        with columns:
        - n: number of observations in the decile
        - defaults: number of observed defaults (sum of `y_true`) in the decile
        - default_rate: defaults / n
        - lift: default_rate divided by overall default rate in `y_true`
        - cum_pct_defaults: cumulative proportion of total defaults captured up to that decile

    Notes
    -----
    - Predictions are binned into 10 quantile-based deciles using `pd.qcut`.
    - Decile numbering is flipped so that lower numbers indicate higher risk (decile 1 = riskiest).
    """
    
    df = pd.DataFrame({"y_true": y_true.values, "pd": y_pred})

    # 1. Bin into deciles. labels=False gives integer codes 0–9 (0 = lowest PD).
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), q=10, labels=False)

    # 2. Flip so decile 1 = riskiest (highest PD), decile 10 = safest.
    df["decile"] = 10 - df["decile"]

    # 3. Aggregate per decile.
    gains = df.groupby("decile").agg(
        n=("y_true", "size"),
        defaults=("y_true", "sum"),
    ).sort_index()

    # 4. Derived columns.
    gains["default_rate"] = gains["defaults"] / gains["n"]
    gains["lift"] = gains["default_rate"] / df["y_true"].mean()
    gains["cum_pct_defaults"] = gains["defaults"].cumsum() / gains["defaults"].sum()

    return gains