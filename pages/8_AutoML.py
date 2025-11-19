import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.automl import run_automl, detect_task_type
from utils.explainability import plot_feature_importance, SHAP_AVAILABLE
import warnings
warnings.filterwarnings('ignore')

st.title("⚡ AutoML - Automated Machine Learning")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload a dataset first.")
else:
    st.markdown("""
    **AutoML automatically:**
    1. Detects whether your task is classification or regression
    2. Trains multiple machine learning models
    3. Evaluates and compares their performance
    4. Recommends the best model for your data

    Just select your target variable and let AutoML do the rest!
    """)

    st.markdown("---")

    # Select target variable
    target_col = st.selectbox(
        "Select Target Variable (Y)",
        df.columns,
        help="The variable you want to predict"
    )

    # Preview target distribution
    st.subheader("Target Variable Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Data Type:** {df[target_col].dtype}")
        st.write(f"**Missing Values:** {df[target_col].isna().sum()}")
        st.write(f"**Unique Values:** {df[target_col].nunique()}")

    with col2:
        # Detect task type
        detected_task = detect_task_type(df[target_col])
        st.info(f"**Detected Task Type:** {detected_task.upper()}")

        if detected_task == 'classification':
            st.write(f"**Classes:** {df[target_col].unique()[:10]}")
        else:
            st.write(f"**Range:** {df[target_col].min():.2f} to {df[target_col].max():.2f}")

    # Distribution plot
    if detected_task == 'classification':
        fig = px.histogram(df, x=target_col, title=f'Distribution of {target_col}')
    else:
        fig = px.histogram(df, x=target_col, title=f'Distribution of {target_col}', nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AutoML Configuration
    st.subheader("AutoML Configuration")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)

    # Feature selection
    feature_selection = st.radio(
        "Feature Selection",
        ["Use all features (automatic)", "Select features manually"]
    )

    if feature_selection == "Select features manually":
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Select Features",
            available_features,
            default=available_features
        )

        if selected_features:
            df_automl = df[[target_col] + selected_features].copy()
        else:
            st.warning("Please select at least one feature.")
            df_automl = None
    else:
        df_automl = df.copy()

    st.markdown("---")

    # Run AutoML button
    if df_automl is not None and st.button("🚀 Run AutoML", key="run_automl"):
        with st.spinner("Running AutoML... This may take a minute..."):
            try:
                # Run AutoML
                task_type, results_df, best_model_name, best_model, X_test, y_test = run_automl(
                    df_automl, target_col, test_size=test_size, random_state=random_state
                )

                st.success(f"✓ AutoML completed! Task Type: {task_type.upper()}")

                # Display results
                st.markdown("---")
                st.subheader("📊 Model Comparison")

                # Results table
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

                # Best model highlight
                st.markdown(f"### 🏆 Best Model: **{best_model_name}**")

                # Metrics visualization
                st.subheader("Model Performance Comparison")

                if task_type == 'classification':
                    # Classification metrics bar chart
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    fig = go.Figure()

                    for metric in metrics:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Model'],
                            y=results_df[metric]
                        ))

                    fig.update_layout(
                        title='Classification Metrics Comparison',
                        xaxis_title='Model',
                        yaxis_title='Score',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # CV Score comparison
                    fig_cv = px.bar(
                        results_df,
                        x='Model',
                        y='CV Score',
                        title='Cross-Validation Score Comparison',
                        color='CV Score',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)

                else:  # Regression
                    # Regression metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        fig_r2 = px.bar(
                            results_df,
                            x='Model',
                            y='R² Score',
                            title='R² Score Comparison',
                            color='R² Score',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_r2, use_container_width=True)

                    with col2:
                        fig_rmse = px.bar(
                            results_df,
                            x='Model',
                            y='RMSE',
                            title='RMSE Comparison (Lower is Better)',
                            color='RMSE',
                            color_continuous_scale='Reds_r'
                        )
                        st.plotly_chart(fig_rmse, use_container_width=True)

                # Feature importance for best model
                if hasattr(best_model, 'feature_importances_'):
                    st.markdown("---")
                    st.subheader(f"Feature Importance - {best_model_name}")

                    feature_names = X_test.columns.tolist()
                    importances = best_model.feature_importances_

                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    fig_imp = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'Top 15 Feature Importances - {best_model_name}',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_imp.update_yaxis(autorange="reversed")
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # Download feature importance
                    csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="Download Feature Importance CSV",
                        data=csv,
                        file_name="feature_importance.csv",
                        mime="text/csv"
                    )

                # Model predictions visualization
                st.markdown("---")
                st.subheader("Model Predictions")

                predictions = best_model.predict(X_test)

                if task_type == 'classification':
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, predictions)

                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"),
                        title=f"Confusion Matrix - {best_model_name}"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                else:  # Regression
                    # Actual vs Predicted
                    results_pred_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': predictions
                    })

                    fig_scatter = px.scatter(
                        results_pred_df,
                        x='Actual',
                        y='Predicted',
                        title=f'Actual vs Predicted - {best_model_name}',
                        trendline='ols'
                    )

                    # Add perfect prediction line
                    fig_scatter.add_trace(go.Scatter(
                        x=[results_pred_df['Actual'].min(), results_pred_df['Actual'].max()],
                        y=[results_pred_df['Actual'].min(), results_pred_df['Actual'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))

                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # Residuals plot
                    residuals = y_test - predictions
                    fig_residuals = px.scatter(
                        x=predictions,
                        y=residuals,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title='Residual Plot'
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_residuals, use_container_width=True)

                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")

                if task_type == 'classification':
                    best_accuracy = results_df.iloc[0]['Accuracy']

                    if best_accuracy > 0.9:
                        st.success("Excellent model performance! Your model is highly accurate.")
                    elif best_accuracy > 0.8:
                        st.info("Good model performance. Consider feature engineering or hyperparameter tuning for improvement.")
                    else:
                        st.warning("Model performance can be improved. Consider:\n- Collecting more data\n- Feature engineering\n- Handling class imbalance\n- Trying different algorithms")

                else:
                    best_r2 = results_df.iloc[0]['R² Score']

                    if best_r2 > 0.8:
                        st.success("Excellent model fit! Your model explains most of the variance in the data.")
                    elif best_r2 > 0.6:
                        st.info("Good model fit. Consider feature engineering or polynomial features for improvement.")
                    else:
                        st.warning("Model fit can be improved. Consider:\n- Adding more relevant features\n- Polynomial or interaction features\n- Checking for outliers\n- Trying ensemble methods")

                # Store best model in session state
                st.session_state['automl_best_model'] = best_model
                st.session_state['automl_best_model_name'] = best_model_name
                st.session_state['automl_task_type'] = task_type

                st.success("✓ Best model saved in session for future use!")

            except Exception as e:
                st.error(f"Error running AutoML: {str(e)}")
                st.info("Please make sure:\n- Your target variable is suitable for prediction\n- You have enough data samples\n- Features contain relevant information")
                import traceback
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

st.success("AutoML page loaded successfully!")
