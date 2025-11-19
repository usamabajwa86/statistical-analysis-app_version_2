import streamlit as st
import pandas as pd
import statsmodels.api as sm
from utils.statistics import run_linear_regression, run_anova
from utils.interpretation import generic_interpretation

st.title("📈 Regression & ANOVA")

df = st.session_state.get("df")

if df is None:
    st.warning("Upload dataset first.")
else:
    analysis_type = st.selectbox("Select Analysis", ["Linear Regression", "ANOVA"])

    if analysis_type == "Linear Regression":
        col_y = st.selectbox("Dependent Variable (Y)", df.columns)
        col_x = st.multiselect("Independent Variables (X)", df.columns)

        if st.button("Run Regression"):
            if col_x:
                result = run_linear_regression(df, col_y, col_x)
                st.write(result.summary())
            else:
                st.warning("Select at least one independent variable.")

    if analysis_type == "ANOVA":
        factor = st.selectbox("Factor Column", df.columns)
        response = st.selectbox("Response Column", df.columns)

        if st.button("Run ANOVA"):
            anova_result = run_anova(df, response, factor)
            st.write(anova_result)
            st.markdown(generic_interpretation("ANOVA", None))

st.success("Regression & ANOVA page loaded successfully!")
