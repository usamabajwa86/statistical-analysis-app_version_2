import streamlit as st
import pandas as pd

st.title("📁 Upload Your Dataset")

uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File uploaded successfully!")
    st.dataframe(df)
    st.session_state["df"] = df

