from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def run_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(df)
    return pca, comps

def run_cca(X, Y, n_components=2):
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    return cca, X_c, Y_c
# utils/advanced_stats.py

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
from sklearn.cross_decomposition import CCA

# -------------------------
# Assumption checks
# -------------------------
def shapiro_test(series):
    # returns stat, p
    clean = series.dropna()
    if clean.shape[0] < 3:
        return None, None
    return stats.shapiro(clean)

def levene_test(df, dv, factor):
    # df: DataFrame, dv: numeric column, factor: grouping var name
    groups = [grp[dv].dropna().values for name, grp in df.groupby(factor)]
    if len(groups) < 2:
        return None, None
    return stats.levene(*groups)

# -------------------------
# One-way ANOVA + Tukey
# -------------------------
def one_way_anova(df, dv, factor):
    model = ols(f"{dv} ~ C({factor})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Tukey requires no NaNs in dv/factor
    tukey = None
    try:
        tukey = pairwise_tukeyhsd(df[dv].dropna(), df[factor].dropna())
    except Exception:
        tukey = None
    return anova_table, tukey

# -------------------------
# Two-way ANOVA (with interaction)
# -------------------------
def two_way_anova(df, dv, factor1, factor2):
    formula = f"{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model

# -------------------------
# Effect sizes (eta-squared)
# -------------------------
def eta_squared(anova_table):
    # anova_table from statsmodels ANOVA LM
    ss_total = anova_table['sum_sq'].sum()
    eta = anova_table['sum_sq'] / ss_total
    return eta

# -------------------------
# MANOVA
# -------------------------
def run_manova(df, dep_vars, indep):
    # dep_vars: list of column names (numeric)
    # indep: string, independent variable (categorical)
    dep_str = '+'.join(dep_vars)
    formula = f"{dep_str} ~ {indep}"
    mv = MANOVA.from_formula(formula, data=df)
    # mv.mv_test() returns a nested results object; convert to string for display
    try:
        res = mv.mv_test()
        return res
    except Exception as e:
        return str(e)

# -------------------------
# CCA
# -------------------------
def run_cca(X_df, Y_df, n_components=2):
    # X_df, Y_df: numeric DataFrames (no NaNs)
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X_df.values, Y_df.values)
    # return canonical variates as DataFrames for plotting
    Xc_df = pd.DataFrame(X_c, columns=[f"CanX{i+1}" for i in range(X_c.shape[1])], index=X_df.index)
    Yc_df = pd.DataFrame(Y_c, columns=[f"CanY{i+1}" for i in range(Y_c.shape[1])], index=Y_df.index)
    return cca, Xc_df, Yc_df

# -------------------------
# Helpers
# -------------------------
def prepare_numeric_df(df, cols):
    return df[cols].apply(pd.to_numeric, errors='coerce')
