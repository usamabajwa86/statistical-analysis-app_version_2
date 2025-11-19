import streamlit as st
import pandas as pd
import statsmodels.api as sm
from utils.statistics import run_linear_regression, run_anova, run_manova
from utils.interpretation import generic_interpretation

st.title("📈 Regression & ANOVA")

df = st.session_state.get("df")

if df is None:
    st.warning("Upload dataset first.")
else:
    analysis_type = st.selectbox("Select Analysis", ["Linear Regression", "ANOVA", "MANOVA"])

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

    if analysis_type == "MANOVA":
        st.markdown("### Multivariate Analysis of Variance (MANOVA)")
        st.info("MANOVA tests the effect of one or more categorical independent variables on multiple dependent variables simultaneously.")

        factor = st.selectbox("Factor Column (Grouping Variable)", df.columns)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        dependent_vars = st.multiselect("Dependent Variables (Select 2 or more)", numeric_cols)

        if st.button("Run MANOVA"):
            if len(dependent_vars) < 2:
                st.warning("Please select at least 2 dependent variables for MANOVA.")
            else:
                try:
                    manova_result = run_manova(df, dependent_vars, factor)
                    st.subheader("MANOVA Results")
                    st.write(manova_result)

                    st.markdown("""
                    **Interpretation Guide:**
                    - **Wilks' Lambda**: Most commonly reported; values closer to 0 indicate stronger effects
                    - **Pillai's Trace**: More robust to violations of assumptions
                    - **Hotelling-Lawley Trace**: More powerful when assumptions are met
                    - **Roy's Greatest Root**: Tests only the largest eigenvalue

                    Look at the **Pr > F** column: p-values < 0.05 typically indicate significant multivariate effects.
                    """)
                except Exception as e:
                    st.error(f"Error running MANOVA: {str(e)}")
                    st.info("Make sure your factor column has at least 2 groups and dependent variables are numeric.")

st.success("Regression & ANOVA page loaded successfully!")
