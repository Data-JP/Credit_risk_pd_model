# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:00:46 2026

@author: J-P-F
"""

"""Feature engineering utilities: Weight of Evidence and Information Value."""
import numpy as np
import pandas as pd

def woe_iv(df, feature, target):
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for a feature.

    For each category of `feature`, compares its share of good (target=0) vs
    bad (target=1) cases. WOE = log(dist_good / dist_bad) per category;
    IV = (dist_good - dist_bad) * WOE per category, summed into Total_IV to
    gauge how predictive the feature is overall.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing `feature` and `target` columns.
    feature : str
        Column to bucket and evaluate (categorical, or already binned).
    target : str, default "SeriousDlqin2yrs"
        Binary target column (0 = good, 1 = bad).

    Returns
    -------
    pd.DataFrame
        One row per category of `feature` with N, Dist_Good, Dist_Bad, WOE,
        IV, and Total_IV (the feature's overall IV, repeated on every row).
    """
    n_good = (df[target] == 0).sum()
    n_bad  = (df[target] == 1).sum()
    rows   = []
    for cat in df[feature].unique():
        g = ((df[feature]==cat) & (df[target]==0)).sum()
        b = ((df[feature]==cat) & (df[target]==1)).sum()
        dist_g = g / n_good if n_good > 0 else 0
        dist_b = b / n_bad  if n_bad > 0  else 0
        woe    = np.log(dist_g / dist_b) if (dist_g > 0 and dist_b > 0) else 0
        iv     = (dist_g - dist_b) * woe
        rows.append({'Category': cat, 'N': (df[feature]==cat).sum(),
                     'Dist_Good': dist_g, 'Dist_Bad': dist_b,
                     'WOE': woe, 'IV': iv})
    result = pd.DataFrame(rows)
    result['Total_IV'] = result['IV'].sum()
    return result