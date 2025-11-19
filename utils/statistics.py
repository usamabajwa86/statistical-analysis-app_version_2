import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------- T-Test ----------
def run_ttest(df, col1, col2):
    stat, p = stats.ttest_ind(df[col1], df[col2], nan_policy='omit')
    return stat, p

# ---------- One-Way ANOVA ----------
def one_way_anova(df, dv, factor):
    model = ols(f"{dv} ~ C({factor})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    tukey = pairwise_tukeyhsd(df[dv], df[factor])
    return anova_table, tukey.summary()

# ---------- Chi-Square ----------
def chi_square_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return stat, p, dof, expected

# ---------- Linear Regression ----------
def run_linear_regression(df, col_y, col_x):
    """
    Runs OLS linear regression with col_y as dependent variable
    and col_x (list) as independent variables.
    """
    X = df[col_x].dropna()
    y = df[col_y].loc[X.index]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# ---------- ANOVA ----------
def run_anova(df, response, factor):
    """
    Runs one-way ANOVA using the specified response and factor columns.
    """
    anova_table, tukey = one_way_anova(df, response, factor)
    return anova_table
