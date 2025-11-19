import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

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


# ---------- AutoML for Classification ----------
def auto_classification(df, target_col, test_size=0.2, random_state=42):
    """
    Automatically trains multiple classification models and selects the best one.

    Parameters:
    - df: DataFrame with features and target
    - target_col: Name of target column
    - test_size: Proportion of test data
    - random_state: Random state for reproducibility

    Returns:
    - results_df: DataFrame with model performance comparison
    - best_model_name: Name of the best performing model
    - best_model: The best trained model
    - X_test, y_test: Test data for further evaluation
    """
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical variables in features
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'SVM': SVC(random_state=random_state)
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss')

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)

    # Train and evaluate models
    results = []
    trained_models = {}

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Cross-validation score
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'CV Score': cv_score
            })
        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    # Get best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]

    return results_df, best_model_name, best_model, X_test, y_test


# ---------- AutoML for Regression ----------
def auto_regression(df, target_col, test_size=0.2, random_state=42):
    """
    Automatically trains multiple regression models and selects the best one.

    Parameters:
    - df: DataFrame with features and target
    - target_col: Name of target column
    - test_size: Proportion of test data
    - random_state: Random state for reproducibility

    Returns:
    - results_df: DataFrame with model performance comparison
    - best_model_name: Name of the best performing model
    - best_model: The best trained model
    - X_test, y_test: Test data for further evaluation
    """
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical variables in features
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=random_state),
        'Lasso Regression': Lasso(random_state=random_state),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'SVR': SVR()
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=random_state)

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1)

    # Train and evaluate models
    results = []
    trained_models = {}

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Cross-validation score
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

            results.append({
                'Model': name,
                'R² Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV R² Score': cv_score
            })
        except Exception as e:
            print(f"Error training {name}: {str(e)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False).reset_index(drop=True)

    # Get best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]

    return results_df, best_model_name, best_model, X_test, y_test


# ---------- Detect Task Type ----------
def detect_task_type(y):
    """
    Automatically detects if the task is classification or regression.

    Parameters:
    - y: Target variable (Series or array)

    Returns:
    - task_type: 'classification' or 'regression'
    """
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(y):
        # Check number of unique values
        unique_ratio = len(y.unique()) / len(y)

        # If less than 10 unique values or less than 5% unique ratio, consider classification
        if len(y.unique()) < 10 or unique_ratio < 0.05:
            return 'classification'
        else:
            return 'regression'
    else:
        # Non-numeric targets are classification
        return 'classification'


# ---------- Full AutoML Pipeline ----------
def run_automl(df, target_col, test_size=0.2, random_state=42):
    """
    Runs full AutoML pipeline: automatically detects task type and trains best model.

    Parameters:
    - df: DataFrame with features and target
    - target_col: Name of target column
    - test_size: Proportion of test data
    - random_state: Random state for reproducibility

    Returns:
    - task_type: 'classification' or 'regression'
    - results_df: Model performance comparison
    - best_model_name: Name of best model
    - best_model: Best trained model
    - X_test, y_test: Test data
    """
    # Detect task type
    task_type = detect_task_type(df[target_col])

    # Run appropriate AutoML
    if task_type == 'classification':
        results_df, best_model_name, best_model, X_test, y_test = auto_classification(
            df, target_col, test_size, random_state
        )
    else:
        results_df, best_model_name, best_model, X_test, y_test = auto_regression(
            df, target_col, test_size, random_state
        )

    return task_type, results_df, best_model_name, best_model, X_test, y_test
