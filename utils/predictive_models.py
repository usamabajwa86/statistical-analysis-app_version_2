import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


# ---------- Logistic Regression ----------
def run_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Runs logistic regression for binary/multi-class classification.

    Returns:
    - model: Trained LogisticRegression model
    - predictions: Predictions on test set
    - metrics: Dictionary with accuracy, confusion matrix, classification report
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions),
        'classification_report': classification_report(y_test, predictions),
        'probabilities': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    }

    return model, predictions, metrics


# ---------- Random Forest ----------
def run_random_forest(X_train, X_test, y_train, y_test, task='classification', n_estimators=100):
    """
    Runs Random Forest for classification or regression.

    Parameters:
    - task: 'classification' or 'regression'
    """
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }
    else:
        metrics = {
            'r2_score': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

    return model, predictions, metrics


# ---------- Gradient Boosting ----------
def run_gradient_boosting(X_train, X_test, y_train, y_test, task='classification', n_estimators=100):
    """
    Runs Gradient Boosting for classification or regression.
    """
    if task == 'classification':
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }
    else:
        metrics = {
            'r2_score': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

    return model, predictions, metrics


# ---------- XGBoost ----------
def run_xgboost(X_train, X_test, y_train, y_test, task='classification', n_estimators=100):
    """
    Runs XGBoost for classification or regression (if available).
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    if task == 'classification':
        model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42, eval_metric='logloss')
    else:
        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }
    else:
        metrics = {
            'r2_score': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

    return model, predictions, metrics


# ---------- LightGBM ----------
def run_lightgbm(X_train, X_test, y_train, y_test, task='classification', n_estimators=100):
    """
    Runs LightGBM for classification or regression (if available).
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=42, verbose=-1)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }
    else:
        metrics = {
            'r2_score': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

    return model, predictions, metrics


# ---------- ARIMA Time Series ----------
def run_arima(data, order=(1, 1, 1), forecast_steps=10):
    """
    Runs ARIMA model for time series forecasting.

    Parameters:
    - data: 1D array or Series of time series data
    - order: (p, d, q) ARIMA parameters
    - forecast_steps: Number of future steps to predict
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=forecast_steps)

    return fitted_model, forecast


# ---------- Prophet Time Series ----------
def run_prophet(df, date_col, value_col, forecast_periods=30):
    """
    Runs Facebook Prophet for time series forecasting.

    Parameters:
    - df: DataFrame with date and value columns
    - date_col: Name of date column
    - value_col: Name of value column
    - forecast_periods: Number of future periods to predict
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Install with: pip install prophet")

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})

    model = Prophet()
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    return model, forecast


# ---------- Data Preparation Helper ----------
def prepare_train_test_split(df, target_col, test_size=0.2, random_state=42):
    """
    Prepares train-test split for modeling.

    Parameters:
    - df: DataFrame
    - target_col: Name of target column
    - test_size: Proportion of test data
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
