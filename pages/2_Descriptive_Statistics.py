import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

st.title("🔍 Descriptive Statistics")

# Check if dataset exists
if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
else:
    df = st.session_state["df"]

    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include='object').columns))

    # First rows
    st.markdown("---")
    st.subheader("Data Preview")
    n_rows = st.slider("Number of rows to display", 5, 50, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # Descriptive Statistics
    st.markdown("---")
    st.subheader("Descriptive Statistics Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Missing Values Analysis
    st.markdown("---")
    st.subheader("Missing Values Analysis")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)

    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)

        # Visualize missing values
        fig_missing = px.bar(missing_df, x='Column', y='Missing %',
                            title='Missing Values by Column',
                            labels={'Missing %': 'Missing Percentage (%)'},
                            color='Missing %',
                            color_continuous_scale='Reds')
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")

    # Correlation Heatmap
    st.markdown("---")
    st.subheader("Correlation Analysis")
    st.info("Explore relationships between numeric variables")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) >= 2:
        # Select variables for correlation
        corr_method = st.selectbox("Correlation Method", ["Pearson", "Spearman", "Kendall"])

        selected_cols = st.multiselect(
            "Select columns for correlation (leave empty for all numeric columns)",
            numeric_cols,
            default=[]
        )

        if not selected_cols:
            selected_cols = numeric_cols

        if st.button("Generate Correlation Heatmap"):
            # Calculate correlation
            if corr_method == "Pearson":
                corr_matrix = df[selected_cols].corr(method='pearson')
            elif corr_method == "Spearman":
                corr_matrix = df[selected_cols].corr(method='spearman')
            else:
                corr_matrix = df[selected_cols].corr(method='kendall')

            # Create heatmap
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title=f'{corr_method} Correlation Heatmap',
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(width=800, height=800)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Strong correlations
            st.markdown("### Strong Correlations (|r| > 0.7)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })

            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', ascending=False, key=abs)
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info("No strong correlations (|r| > 0.7) found")

    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

    # Distribution Analysis
    st.markdown("---")
    st.subheader("Distribution Analysis")

    if numeric_cols:
        selected_var = st.selectbox("Select variable to analyze", numeric_cols)

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist = px.histogram(df, x=selected_var, nbins=30,
                                   title=f'Distribution of {selected_var}',
                                   marginal='box')
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Box plot
            fig_box = px.box(df, y=selected_var,
                            title=f'Box Plot of {selected_var}')
            st.plotly_chart(fig_box, use_container_width=True)

        # Statistical tests
        st.markdown("#### Normality Test")
        data = df[selected_var].dropna()

        if len(data) > 3:
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shapiro-Wilk Statistic", f"{statistic:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")

            if p_value > 0.05:
                st.success(f"Data appears to be normally distributed (p > 0.05)")
            else:
                st.warning(f"Data does not appear to be normally distributed (p ≤ 0.05)")

    # Categorical Variables Analysis
    if df.select_dtypes(include='object').shape[1] > 0:
        st.markdown("---")
        st.subheader("Categorical Variables Analysis")

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        selected_cat = st.selectbox("Select categorical variable", cat_cols)

        # Value counts
        value_counts = df[selected_cat].value_counts().head(20)

        fig_cat = px.bar(x=value_counts.index, y=value_counts.values,
                        labels={'x': selected_cat, 'y': 'Count'},
                        title=f'Top Categories in {selected_cat}')
        st.plotly_chart(fig_cat, use_container_width=True)

        # Summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Categories", df[selected_cat].nunique())
        with col2:
            st.metric("Most Common", df[selected_cat].mode()[0] if len(df[selected_cat].mode()) > 0 else "N/A")

st.success("Descriptive Statistics page loaded successfully!")

