import streamlit as st
import pandas as pd

st.title("🔍 Descriptive Statistics")

# Check if dataset exists
if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
else:
    df = st.session_state["df"]
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    st.write("Descriptive statistics summary:")
    st.write(df.describe())

