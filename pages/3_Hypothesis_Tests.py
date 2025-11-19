import streamlit as st
from utils.statistics import run_ttest, chi_square_test
from utils.interpretation import generic_interpretation

st.title("📊 Hypothesis Tests")

df = st.session_state.get("df")

if df is None:
    st.warning("Upload dataset first.")
else:
    test = st.selectbox("Select Test", ["T-Test", "Chi-Square"])

    if test == "T-Test":
        col1 = st.selectbox("Column 1", df.columns)
        col2 = st.selectbox("Column 2", df.columns)

        if st.button("Run T-Test"):
            stat, p = run_ttest(df, col1, col2)
            st.write("Statistic:", stat)
            st.write("P-value:", p)
            st.markdown(generic_interpretation("T-test", p))

    if test == "Chi-Square":
        col1 = st.selectbox("Categorical Column 1", df.columns)
        col2 = st.selectbox("Categorical Column 2", df.columns)

        if st.button("Run Chi-Square"):
            stat, p, dof, exp = chi_square_test(df, col1, col2)
            st.write("Statistic:", stat)
            st.write("P-value:", p)
            st.markdown(generic_interpretation("Chi-Square", p))

st.success("Hypothesis tests page loaded successful")
