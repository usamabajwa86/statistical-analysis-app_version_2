import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from utils.clustering import (
    run_kmeans, run_hierarchical_clustering, run_dbscan,
    run_tsne, run_umap, calculate_elbow_scores, UMAP_AVAILABLE
)

st.title("🔍 Clustering & Segmentation")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload a dataset first.")
else:
    st.markdown("""
    Discover patterns and group similar data points using unsupervised learning.
    Identify clusters, segments, and outliers in your data.
    """)

    # Select clustering algorithm
    algorithm = st.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "Hierarchical Clustering", "DBSCAN (Outlier Detection)"]
    )

    # Select features
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_features = st.multiselect(
        "Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )

    if not selected_features:
        st.warning("Please select at least 2 features for clustering.")
    elif len(selected_features) < 2:
        st.warning("Please select at least 2 features for meaningful clustering.")
    else:
        df_cluster = df[selected_features].dropna()

        # ========== K-MEANS ==========
        if algorithm == "K-Means":
            st.markdown("### K-Means Clustering")
            st.info("Groups data into K distinct clusters based on similarity")

            col1, col2 = st.columns(2)
            with col1:
                show_elbow = st.checkbox("Show Elbow Plot (Find Optimal K)", value=False)
            with col2:
                n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

            scale_data = st.checkbox("Standardize features (recommended)", value=True)

            if show_elbow:
                with st.spinner("Calculating elbow scores..."):
                    try:
                        scores = calculate_elbow_scores(df_cluster, max_k=10, scale=scale_data)

                        # Plot elbow
                        fig_elbow = go.Figure()
                        fig_elbow.add_trace(go.Scatter(
                            x=scores['k_values'], y=scores['inertias'],
                            mode='lines+markers', name='Inertia'
                        ))
                        fig_elbow.update_layout(
                            title='Elbow Method: Optimal K',
                            xaxis_title='Number of Clusters (K)',
                            yaxis_title='Inertia'
                        )
                        st.plotly_chart(fig_elbow, use_container_width=True)

                        # Plot silhouette scores
                        fig_sil = go.Figure()
                        fig_sil.add_trace(go.Scatter(
                            x=scores['k_values'], y=scores['silhouette_scores'],
                            mode='lines+markers', name='Silhouette Score', line_color='green'
                        ))
                        fig_sil.update_layout(
                            title='Silhouette Score by K',
                            xaxis_title='Number of Clusters (K)',
                            yaxis_title='Silhouette Score'
                        )
                        st.plotly_chart(fig_sil, use_container_width=True)

                        st.info("Look for the 'elbow' point where inertia starts decreasing more slowly, or the K with highest silhouette score.")

                    except Exception as e:
                        st.error(f"Error calculating elbow scores: {str(e)}")

            if st.button("Run K-Means Clustering"):
                with st.spinner("Running K-Means..."):
                    try:
                        model, labels, metrics = run_kmeans(
                            df_cluster, n_clusters=n_clusters, scale=scale_data
                        )

                        st.success("✓ K-Means clustering completed!")

                        # Display metrics
                        st.subheader("Clustering Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
                        with col2:
                            st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.4f}")
                        with col3:
                            st.metric("Inertia", f"{metrics['inertia']:.2f}")

                        st.info("Higher Silhouette Score is better. Lower Davies-Bouldin Score is better.")

                        # Add cluster labels to data
                        df_cluster_with_labels = df_cluster.copy()
                        df_cluster_with_labels['Cluster'] = labels

                        # Visualize clusters
                        st.subheader("Cluster Visualization")

                        # 2D scatter plot
                        if len(selected_features) >= 2:
                            fig = px.scatter(
                                df_cluster_with_labels,
                                x=selected_features[0],
                                y=selected_features[1],
                                color='Cluster',
                                title=f'Clusters: {selected_features[0]} vs {selected_features[1]}',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Cluster sizes
                        st.subheader("Cluster Distribution")
                        cluster_counts = pd.Series(labels).value_counts().sort_index()
                        fig_bar = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                        labels={'x': 'Cluster', 'y': 'Number of Samples'},
                                        title='Cluster Sizes')
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        cluster_stats = df_cluster_with_labels.groupby('Cluster')[selected_features].mean()
                        st.dataframe(cluster_stats)

                    except Exception as e:
                        st.error(f"Error running K-Means: {str(e)}")

        # ========== HIERARCHICAL CLUSTERING ==========
        elif algorithm == "Hierarchical Clustering":
            st.markdown("### Hierarchical Clustering")
            st.info("Creates a tree-like hierarchy of clusters")

            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            with col2:
                linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])

            scale_data = st.checkbox("Standardize features (recommended)", value=True)

            if st.button("Run Hierarchical Clustering"):
                with st.spinner("Running Hierarchical Clustering..."):
                    try:
                        model, labels, linkage_matrix, metrics = run_hierarchical_clustering(
                            df_cluster, n_clusters=n_clusters, linkage_method=linkage_method, scale=scale_data
                        )

                        st.success("✓ Hierarchical clustering completed!")

                        # Display metrics
                        st.subheader("Clustering Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
                        with col2:
                            st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.4f}")

                        # Dendrogram
                        st.subheader("Dendrogram")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(12, 6))
                        dendrogram(linkage_matrix, ax=ax)
                        ax.set_title('Hierarchical Clustering Dendrogram')
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Distance')
                        st.pyplot(fig)

                        # Add cluster labels to data
                        df_cluster_with_labels = df_cluster.copy()
                        df_cluster_with_labels['Cluster'] = labels

                        # Visualize clusters
                        st.subheader("Cluster Visualization")
                        if len(selected_features) >= 2:
                            fig = px.scatter(
                                df_cluster_with_labels,
                                x=selected_features[0],
                                y=selected_features[1],
                                color='Cluster',
                                title=f'Clusters: {selected_features[0]} vs {selected_features[1]}',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        cluster_stats = df_cluster_with_labels.groupby('Cluster')[selected_features].mean()
                        st.dataframe(cluster_stats)

                    except Exception as e:
                        st.error(f"Error running Hierarchical Clustering: {str(e)}")

        # ========== DBSCAN ==========
        elif algorithm == "DBSCAN (Outlier Detection)":
            st.markdown("### DBSCAN - Density-Based Clustering")
            st.info("Identifies dense clusters and detects outliers/noise points (labeled as -1)")

            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("Epsilon (neighborhood radius)", 0.1, 2.0, 0.5, 0.1)
            with col2:
                min_samples = st.slider("Minimum Samples", 2, 20, 5)

            scale_data = st.checkbox("Standardize features (recommended)", value=True)

            st.markdown(f"""
            **Parameters:**
            - **Epsilon ({eps})**: Maximum distance between two samples to be considered neighbors
            - **Min Samples ({min_samples})**: Minimum number of samples in a neighborhood for a core point
            """)

            if st.button("Run DBSCAN"):
                with st.spinner("Running DBSCAN..."):
                    try:
                        model, labels, metrics = run_dbscan(
                            df_cluster, eps=eps, min_samples=min_samples, scale=scale_data
                        )

                        st.success("✓ DBSCAN completed!")

                        # Display metrics
                        st.subheader("Clustering Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Number of Clusters", metrics['n_clusters'])
                        with col2:
                            st.metric("Noise Points", metrics['n_noise_points'])
                        with col3:
                            st.metric("Noise %", f"{metrics['noise_percentage']:.1f}%")

                        if 'silhouette_score' in metrics:
                            st.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")

                        # Add cluster labels to data
                        df_cluster_with_labels = df_cluster.copy()
                        df_cluster_with_labels['Cluster'] = labels

                        # Visualize clusters
                        st.subheader("Cluster Visualization")
                        if len(selected_features) >= 2:
                            fig = px.scatter(
                                df_cluster_with_labels,
                                x=selected_features[0],
                                y=selected_features[1],
                                color='Cluster',
                                title=f'DBSCAN Clusters (Cluster -1 = Outliers)',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Cluster sizes
                        st.subheader("Cluster Distribution")
                        cluster_counts = pd.Series(labels).value_counts().sort_index()
                        fig_bar = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                        labels={'x': 'Cluster (-1 = Noise)', 'y': 'Number of Samples'},
                                        title='Cluster Sizes')
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Outlier details
                        if metrics['n_noise_points'] > 0:
                            st.subheader("Outlier/Noise Points")
                            outliers = df_cluster_with_labels[df_cluster_with_labels['Cluster'] == -1]
                            st.write(f"Found {len(outliers)} outlier points:")
                            st.dataframe(outliers)

                    except Exception as e:
                        st.error(f"Error running DBSCAN: {str(e)}")

    # ========== DIMENSIONALITY REDUCTION VISUALIZATION ==========
    st.markdown("---")
    st.markdown("### Dimensionality Reduction Visualization")
    st.info("Visualize high-dimensional data in 2D using t-SNE or UMAP")

    reduction_method = st.selectbox("Select Method", ["t-SNE", "UMAP"] if UMAP_AVAILABLE else ["t-SNE"])

    if len(selected_features) >= 2:
        if reduction_method == "t-SNE":
            perplexity = st.slider("Perplexity", 5, 50, 30, 5)

            if st.button("Run t-SNE"):
                with st.spinner("Running t-SNE..."):
                    try:
                        embedding = run_tsne(df_cluster, n_components=2, perplexity=perplexity)

                        st.success("✓ t-SNE completed!")

                        # Visualize
                        tsne_df = pd.DataFrame(embedding, columns=['t-SNE 1', 't-SNE 2'])
                        fig = px.scatter(tsne_df, x='t-SNE 1', y='t-SNE 2',
                                        title='t-SNE 2D Visualization')
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error running t-SNE: {str(e)}")

        elif reduction_method == "UMAP":
            col1, col2 = st.columns(2)
            with col1:
                n_neighbors = st.slider("Number of Neighbors", 2, 50, 15)
            with col2:
                min_dist = st.slider("Minimum Distance", 0.0, 0.99, 0.1, 0.05)

            if st.button("Run UMAP"):
                with st.spinner("Running UMAP..."):
                    try:
                        embedding = run_umap(df_cluster, n_components=2,
                                           n_neighbors=n_neighbors, min_dist=min_dist)

                        st.success("✓ UMAP completed!")

                        # Visualize
                        umap_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
                        fig = px.scatter(umap_df, x='UMAP 1', y='UMAP 2',
                                        title='UMAP 2D Visualization')
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error running UMAP: {str(e)}")
                        if "not installed" in str(e):
                            st.info("Install UMAP with: pip install umap-learn")

st.success("Clustering & Segmentation page loaded successfully!")
