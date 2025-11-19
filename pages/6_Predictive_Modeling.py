import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.predictive_models import (
    run_logistic_regression, run_random_forest, run_gradient_boosting,
    run_xgboost, run_lightgbm, run_arima, run_prophet, prepare_train_test_split,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, PROPHET_AVAILABLE
)
from utils.explainability import plot_feature_importance, SHAP_AVAILABLE
import warnings
warnings.filterwarnings('ignore')

st.title("🤖 Predictive Modeling")

df = st.session_state.get("df")

if df is None:
    st.warning("Please upload a dataset first.")
else:
    st.markdown("""
    Build and evaluate predictive models using various machine learning algorithms.
    Supports both **classification** and **regression** tasks, plus **time series forecasting**.
    """)

    # Model type selection
    model_category = st.selectbox(
        "Select Model Category",
        ["Classification", "Regression", "Time Series Forecasting"]
    )

    # ========== CLASSIFICATION ==========
    if model_category == "Classification":
        st.markdown("### Classification Models")
        st.info("Predict categorical outcomes (e.g., disease/no disease, success/failure)")

        # Select target and features
        target_col = st.selectbox("Target Variable (Y)", df.columns)
        feature_cols = st.multiselect("Feature Variables (X)",
                                      [col for col in df.columns if col != target_col])

        if not feature_cols:
            st.warning("Please select at least one feature variable.")
        else:
            # Filter data
            df_model = df[[target_col] + feature_cols].dropna()

            # Model selection
            available_models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            if XGBOOST_AVAILABLE:
                available_models.append("XGBoost")
            if LIGHTGBM_AVAILABLE:
                available_models.append("LightGBM")

            model_type = st.selectbox("Select Classification Model", available_models)

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                if model_type in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
                    n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)

            # Train model button
            if st.button("Train Model", key="train_classification"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        X_train, X_test, y_train, y_test = prepare_train_test_split(
                            df_model, target_col, test_size=test_size
                        )

                        # Train selected model
                        if model_type == "Logistic Regression":
                            model, predictions, metrics = run_logistic_regression(
                                X_train, X_test, y_train, y_test
                            )
                        elif model_type == "Random Forest":
                            model, predictions, metrics = run_random_forest(
                                X_train, X_test, y_train, y_test, task='classification', n_estimators=n_estimators
                            )
                        elif model_type == "Gradient Boosting":
                            model, predictions, metrics = run_gradient_boosting(
                                X_train, X_test, y_train, y_test, task='classification', n_estimators=n_estimators
                            )
                        elif model_type == "XGBoost":
                            model, predictions, metrics = run_xgboost(
                                X_train, X_test, y_train, y_test, task='classification', n_estimators=n_estimators
                            )
                        elif model_type == "LightGBM":
                            model, predictions, metrics = run_lightgbm(
                                X_train, X_test, y_train, y_test, task='classification', n_estimators=n_estimators
                            )

                        # Display results
                        st.success(f"✓ Model trained successfully!")

                        # Metrics
                        st.subheader("Model Performance")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        with col2:
                            if metrics.get('probabilities') is not None:
                                st.info(f"Test samples: {len(y_test)}")

                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = metrics['confusion_matrix']
                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                          labels=dict(x="Predicted", y="Actual"),
                                          title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # Classification Report
                        st.subheader("Classification Report")
                        st.text(metrics['classification_report'])

                        # Feature Importance
                        if 'feature_importance' in metrics:
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame(
                                list(metrics['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False)

                            fig_imp = px.bar(importance_df.head(10), x='Importance', y='Feature',
                                           orientation='h', title='Top 10 Feature Importances')
                            fig_imp.update_yaxis(autorange="reversed")
                            st.plotly_chart(fig_imp, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        st.info("Make sure your target variable is categorical and features are numeric.")

    # ========== REGRESSION ==========
    elif model_category == "Regression":
        st.markdown("### Regression Models")
        st.info("Predict continuous outcomes (e.g., crop yield, growth rate, prices)")

        # Select target and features
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        target_col = st.selectbox("Target Variable (Y)", numeric_cols)
        feature_cols = st.multiselect("Feature Variables (X)",
                                      [col for col in numeric_cols if col != target_col])

        if not feature_cols:
            st.warning("Please select at least one feature variable.")
        else:
            # Filter data
            df_model = df[[target_col] + feature_cols].dropna()

            # Model selection
            available_models = ["Random Forest", "Gradient Boosting"]
            if XGBOOST_AVAILABLE:
                available_models.append("XGBoost")
            if LIGHTGBM_AVAILABLE:
                available_models.append("LightGBM")

            model_type = st.selectbox("Select Regression Model", available_models)

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)

            # Train model button
            if st.button("Train Model", key="train_regression"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        X_train, X_test, y_train, y_test = prepare_train_test_split(
                            df_model, target_col, test_size=test_size
                        )

                        # Train selected model
                        if model_type == "Random Forest":
                            model, predictions, metrics = run_random_forest(
                                X_train, X_test, y_train, y_test, task='regression', n_estimators=n_estimators
                            )
                        elif model_type == "Gradient Boosting":
                            model, predictions, metrics = run_gradient_boosting(
                                X_train, X_test, y_train, y_test, task='regression', n_estimators=n_estimators
                            )
                        elif model_type == "XGBoost":
                            model, predictions, metrics = run_xgboost(
                                X_train, X_test, y_train, y_test, task='regression', n_estimators=n_estimators
                            )
                        elif model_type == "LightGBM":
                            model, predictions, metrics = run_lightgbm(
                                X_train, X_test, y_train, y_test, task='regression', n_estimators=n_estimators
                            )

                        # Display results
                        st.success(f"✓ Model trained successfully!")

                        # Metrics
                        st.subheader("Model Performance")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col3:
                            st.info(f"Test samples: {len(y_test)}")

                        # Prediction vs Actual
                        st.subheader("Predictions vs Actual Values")
                        results_df = pd.DataFrame({
                            'Actual': y_test,
                            'Predicted': predictions
                        })

                        fig_scatter = px.scatter(results_df, x='Actual', y='Predicted',
                                               title='Predictions vs Actual',
                                               trendline='ols')
                        fig_scatter.add_trace(go.Scatter(
                            x=[results_df['Actual'].min(), results_df['Actual'].max()],
                            y=[results_df['Actual'].min(), results_df['Actual'].max()],
                            mode='lines', name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        st.plotly_chart(fig_scatter, use_container_width=True)

                        # Feature Importance
                        if 'feature_importance' in metrics:
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame(
                                list(metrics['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False)

                            fig_imp = px.bar(importance_df.head(10), x='Importance', y='Feature',
                                           orientation='h', title='Top 10 Feature Importances')
                            fig_imp.update_yaxis(autorange="reversed")
                            st.plotly_chart(fig_imp, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")

    # ========== TIME SERIES ==========
    elif model_category == "Time Series Forecasting":
        st.markdown("### Time Series Forecasting")
        st.info("Predict future values based on historical time-series data (e.g., crop growth over time)")

        # Select columns
        date_col = st.selectbox("Date/Time Column", df.columns)
        value_col = st.selectbox("Value Column", df.select_dtypes(include='number').columns)

        model_type = st.selectbox("Select Model", ["ARIMA", "Prophet"] if PROPHET_AVAILABLE else ["ARIMA"])

        if model_type == "ARIMA":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                p = st.number_input("p (AR order)", 0, 5, 1)
            with col2:
                d = st.number_input("d (Differencing)", 0, 2, 1)
            with col3:
                q = st.number_input("q (MA order)", 0, 5, 1)
            with col4:
                forecast_steps = st.number_input("Forecast Steps", 1, 100, 10)

            if st.button("Run ARIMA Forecast"):
                with st.spinner("Running ARIMA..."):
                    try:
                        data = df[value_col].dropna()
                        model, forecast = run_arima(data, order=(p, d, q), forecast_steps=forecast_steps)

                        st.success("✓ ARIMA model fitted successfully!")

                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=data, mode='lines', name='Historical'))
                        forecast_index = range(len(data), len(data) + forecast_steps)
                        fig.add_trace(go.Scatter(x=list(forecast_index), y=forecast,
                                                mode='lines', name='Forecast',
                                                line=dict(dash='dash')))
                        fig.update_layout(title='ARIMA Forecast', xaxis_title='Time', yaxis_title='Value')
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Forecasted Values")
                        st.dataframe(pd.DataFrame({'Forecast': forecast}))

                    except Exception as e:
                        st.error(f"Error running ARIMA: {str(e)}")

        elif model_type == "Prophet":
            forecast_periods = st.number_input("Forecast Periods", 1, 365, 30)

            if st.button("Run Prophet Forecast"):
                with st.spinner("Running Prophet..."):
                    try:
                        model, forecast = run_prophet(df, date_col, value_col, forecast_periods)

                        st.success("✓ Prophet model fitted successfully!")

                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col],
                                                mode='markers', name='Actual'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                                mode='lines', name='Forecast'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                                fill=None, mode='lines', line_color='lightblue',
                                                showlegend=False))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                                fill='tonexty', mode='lines', line_color='lightblue',
                                                name='Confidence Interval'))
                        fig.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Value')
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Forecast Data")
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))

                    except Exception as e:
                        st.error(f"Error running Prophet: {str(e)}")

st.success("Predictive Modeling page loaded successfully!")
