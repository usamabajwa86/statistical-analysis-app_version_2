import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA

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

# ---------- MANOVA ----------
def run_manova(df, dependent_vars, factor):
    """
    Runs Multivariate Analysis of Variance (MANOVA) with multiple dependent variables.

    Parameters:
    - df: DataFrame containing the data
    - dependent_vars: List of column names for dependent variables
    - factor: Column name for the grouping factor

    Returns:
    - MANOVA results object
    """
    # Create formula: "dep_var1 + dep_var2 + ... ~ C(factor)"
    dep_formula = " + ".join(dependent_vars)
    formula = f"{dep_formula} ~ C({factor})"

    # Drop rows with missing values in relevant columns
    cols_needed = dependent_vars + [factor]
    df_clean = df[cols_needed].dropna()

    # Run MANOVA
    manova = MANOVA.from_formula(formula, data=df_clean)
    result = manova.mv_test()

    return result
