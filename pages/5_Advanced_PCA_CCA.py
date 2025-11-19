import streamlit as st
from utils.advanced_stats import run_pca, run_cca, prepare_numeric_df
from utils.visualization import pca_scatter
import pandas as pd
import plotly.express as px

st.title("🧪 Advanced PCA & CCA")

df = st.session_state.get("df")

if df is None:
    st.warning("Upload dataset first.")
else:
    st.markdown("## PCA")
    cols = st.multiselect("Select numeric columns for PCA", df.select_dtypes(include="number").columns)

    n_comp = st.slider("Number of PCA components", 2, min(len(cols), 5) if cols else 2, 2)

    if st.button("Run PCA"):
        if len(cols) < 2:
            st.error("Select at least 2 columns for PCA")
        else:
            pca, comps = run_pca(df[cols])
            st.write("Explained Variance:", pca.explained_variance_ratio_)

            fig = pca_scatter(pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_comp)]),
                              x="PC1", y="PC2")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("## CCA")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    left = st.multiselect("Left set (X variables)", numeric_cols, default=numeric_cols[:2])
    right = st.multiselect("Right set (Y variables)", numeric_cols, default=numeric_cols[2:4])
    n_cca = st.slider("Number of canonical components", 1, min(len(left), len(right), 5), 2)

    if st.button("Run CCA"):
        if len(left) < 1 or len(right) < 1:
            st.error("Select at least 1 variable for each set")
        else:
            X_df = prepare_numeric_df(df, left).dropna()
            Y_df = prepare_numeric_df(df, right).dropna()
            common_idx = X_df.index.intersection(Y_df.index)
            X_df = X_df.loc[common_idx]
            Y_df = Y_df.loc[common_idx]

            cca, Xc, Yc = run_cca(X_df, Y_df, n_components=n_cca)
            st.subheader("Canonical Variates (first two shown)")
            st.dataframe(pd.concat([Xc, Yc], axis=1).head())

            fig = px.scatter(pd.concat([Xc.iloc[:, :2], Yc.iloc[:, :2]], axis=1),
                             x=Xc.columns[0], y=Xc.columns[1], title="CCA: Canonical Variates (first 2)")
            st.plotly_chart(fig, use_container_width=True)
