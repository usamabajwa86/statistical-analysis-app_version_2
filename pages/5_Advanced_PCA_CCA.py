import streamlit as st
from utils.advanced_stats import run_pca, run_cca, prepare_numeric_df
from utils.visualization import pca_scatter
from utils.clustering import run_tsne, run_umap, UMAP_AVAILABLE
import pandas as pd
import plotly.express as px

st.title("🧪 Advanced Dimensionality Reduction")

df = st.session_state.get("df")

if df is None:
    st.warning("Upload dataset first.")
else:
    st.markdown("## PCA")
    cols = st.multiselect("Select numeric columns for PCA", df.select_dtypes(include="number").columns)

    max_comp = max(min(len(cols), 5), 2) if cols else 2
    n_comp = st.slider("Number of PCA components", 1, max_comp, min(2, max_comp))

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

    st.markdown("---")
    st.markdown("## t-SNE - t-Distributed Stochastic Neighbor Embedding")
    st.info("Non-linear dimensionality reduction for visualizing high-dimensional data. Great for finding clusters and patterns.")

    tsne_cols = st.multiselect("Select features for t-SNE", df.select_dtypes(include="number").columns, key="tsne_cols")

    if tsne_cols and len(tsne_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            perplexity = st.slider("Perplexity (5-50)", 5, 50, 30, 5,
                                  help="Related to number of nearest neighbors. Higher values = more global structure.")
        with col2:
            tsne_random_state = st.number_input("Random State", 0, 1000, 42, key="tsne_random")

        if st.button("Run t-SNE"):
            with st.spinner("Running t-SNE... This may take a minute..."):
                try:
                    df_tsne = df[tsne_cols].dropna()
                    embedding = run_tsne(df_tsne, n_components=2, perplexity=perplexity, random_state=tsne_random_state)

                    st.success("✓ t-SNE completed!")

                    # Create visualization dataframe
                    tsne_df = pd.DataFrame(embedding, columns=['t-SNE 1', 't-SNE 2'])

                    # Add original index for color coding if needed
                    tsne_df['Sample'] = range(len(tsne_df))

                    fig = px.scatter(tsne_df, x='t-SNE 1', y='t-SNE 2', title='t-SNE 2D Visualization',
                                    hover_data=['Sample'])
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""
                    **Interpretation:**
                    - Points close together are similar in the original high-dimensional space
                    - Clusters indicate groups of similar samples
                    - Distances between clusters are not meaningful
                    """)

                except Exception as e:
                    st.error(f"Error running t-SNE: {str(e)}")
    elif tsne_cols and len(tsne_cols) < 2:
        st.warning("Select at least 2 features for t-SNE")

    st.markdown("---")
    st.markdown("## UMAP - Uniform Manifold Approximation and Projection")
    st.info("Modern alternative to t-SNE. Preserves both local and global structure better than t-SNE.")

    if UMAP_AVAILABLE:
        umap_cols = st.multiselect("Select features for UMAP", df.select_dtypes(include="number").columns, key="umap_cols")

        if umap_cols and len(umap_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_neighbors = st.slider("Number of Neighbors (2-50)", 2, 50, 15,
                                       help="Controls local vs global structure. Higher = more global.")
            with col2:
                min_dist = st.slider("Minimum Distance (0-0.99)", 0.0, 0.99, 0.1, 0.05,
                                    help="How tightly points are packed. Lower = tighter clusters.")
            with col3:
                umap_random_state = st.number_input("Random State", 0, 1000, 42, key="umap_random")

            if st.button("Run UMAP"):
                with st.spinner("Running UMAP..."):
                    try:
                        df_umap = df[umap_cols].dropna()
                        embedding = run_umap(df_umap, n_components=2, n_neighbors=n_neighbors,
                                           min_dist=min_dist, random_state=umap_random_state)

                        st.success("✓ UMAP completed!")

                        # Create visualization dataframe
                        umap_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
                        umap_df['Sample'] = range(len(umap_df))

                        fig = px.scatter(umap_df, x='UMAP 1', y='UMAP 2', title='UMAP 2D Visualization',
                                        hover_data=['Sample'])
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("""
                        **Interpretation:**
                        - Similar to t-SNE but faster and preserves more global structure
                        - Distances between clusters have more meaning than t-SNE
                        - Better for larger datasets
                        """)

                    except Exception as e:
                        st.error(f"Error running UMAP: {str(e)}")
        elif umap_cols and len(umap_cols) < 2:
            st.warning("Select at least 2 features for UMAP")
    else:
        st.warning("UMAP is not installed. Install with: `pip install umap-learn`")

st.success("Advanced Dimensionality Reduction page loaded successfully!")
