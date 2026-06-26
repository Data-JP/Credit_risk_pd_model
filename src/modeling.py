# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:41:06 2026

@author: J-P-F
"""

"""Modeling utilities: feature-ablation refit for logistic regression."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def refit_without_feature(feature_to_drop, features, df, target):
    '''
    Refit a logistic regression model after removing a specified feature.

    This function removes `feature_to_drop` from the provided `features` list,
    re-creates the feature matrix `X`, performs a stratified train/test split,
    standardises the features with `StandardScaler`, fits a logistic regression
    model (with `class_weight='balanced'`), prints the test AUC and the
    model coefficients, and returns the fitted model and related objects.

    Parameters
    ----------
    feature_to_drop : str
        Name of the feature to remove from `features` before refitting.
    features : list of str
        Original list of feature names to consider (one will be removed).
    df : pandas.DataFrame, optional
        Dataset containing the features. Default is the global `credit_data`.
    target : array-like or pandas.Series, optional
        Target values aligned with `df`. Default is the global `y`.

    Returns
    -------
    logreg : sklearn.linear_model.LogisticRegression
        The fitted logistic regression model trained on the reduced feature set.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on the training features (useful to transform new data).
    X_test_slr : numpy.ndarray
        Standardised test features corresponding to `y_test`.
    y_test : pandas.Series or ndarray
        True target values for the test set.

    Notes
    -----
    - The function prints the AUC computed on the test split and a DataFrame of
      coefficients for quick inspection. It uses `random_state=66` and
      `test_size=0.3` for reproducibility and stratification by `target`.
    '''

    # Drop feature(s) to the features list
    keep = [f for f in features if f != feature_to_drop]   # copy, no mutation

    # Re-define X with the reduced feature set
    X = df[keep]

    # Re-split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, random_state=66, test_size=0.3, stratify=target)

    # Z-score standardisation
    scaler = StandardScaler()
    X_train_slr = scaler.fit_transform(X_train)
    X_test_slr = scaler.transform(X_test)

    # Fit logistic regression model on the reduced feature set
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train_slr, y_train)

    # Evaluate the model using ROC AUC score
    pd_logreg = logreg.predict_proba(X_test_slr)[:, 1]  # get probabilities for the positive class
    print(f"AUC after dropping {feature_to_drop}: ", roc_auc_score(y_test, pd_logreg))

    # Display the coefficients of the logistic regression model
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': logreg.coef_[0]
    })

     # return the model, scaler, and feature and their coefficient for later use
    return logreg, scaler, X_train, X_train_slr, X_test, X_test_slr, y_test, pd_logreg, coef_df  
