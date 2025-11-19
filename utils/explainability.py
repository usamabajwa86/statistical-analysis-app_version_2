import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------- Feature Importance Plot ----------
def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots feature importance for tree-based models.

    Parameters:
    - model: Trained model with feature_importances_ attribute
    - feature_names: List of feature names
    - top_n: Number of top features to display

    Returns:
    - fig: Matplotlib figure
    - importance_df: DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    # Get feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Plot top N features
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features['Feature'], top_features['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()

    return fig, importance_df


# ---------- SHAP Summary Plot ----------
def shap_summary_plot(model, X, plot_type='dot'):
    """
    Creates SHAP summary plot for model interpretation.

    Parameters:
    - model: Trained model
    - X: Feature data (DataFrame or array)
    - plot_type: 'dot', 'bar', or 'violin'

    Returns:
    - fig: Matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except:
        try:
            explainer = shap.LinearExplainer(model, X)
        except:
            explainer = shap.KernelExplainer(model.predict, X)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    plt.tight_layout()

    return fig, explainer, shap_values


# ---------- SHAP Waterfall Plot ----------
def shap_waterfall_plot(explainer, shap_values, X, index=0):
    """
    Creates SHAP waterfall plot for a single prediction.

    Parameters:
    - explainer: SHAP explainer object
    - shap_values: SHAP values
    - X: Feature data
    - index: Index of the sample to explain

    Returns:
    - fig: Matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_sample = shap_values[0][index]
    else:
        shap_values_sample = shap_values[index]

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                       else explainer.expected_value[0],
            data=X.iloc[index] if hasattr(X, 'iloc') else X[index]
        ),
        show=False
    )
    plt.tight_layout()

    return fig


# ---------- SHAP Force Plot ----------
def shap_force_plot(explainer, shap_values, X, index=0):
    """
    Creates SHAP force plot for a single prediction.

    Parameters:
    - explainer: SHAP explainer object
    - shap_values: SHAP values
    - X: Feature data
    - index: Index of the sample to explain

    Returns:
    - force_plot: SHAP force plot object (for display in notebooks/streamlit)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_sample = shap_values[0][index]
        base_value = explainer.expected_value[0]
    else:
        shap_values_sample = shap_values[index]
        base_value = explainer.expected_value

    force_plot = shap.force_plot(
        base_value,
        shap_values_sample,
        X.iloc[index] if hasattr(X, 'iloc') else X[index]
    )

    return force_plot


# ---------- SHAP Dependence Plot ----------
def shap_dependence_plot(shap_values, X, feature_name):
    """
    Creates SHAP dependence plot showing how a single feature affects predictions.

    Parameters:
    - shap_values: SHAP values
    - X: Feature data (DataFrame)
    - feature_name: Name of feature to analyze

    Returns:
    - fig: Matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values

    shap.dependence_plot(
        feature_name,
        shap_values_to_plot,
        X,
        show=False
    )
    plt.tight_layout()

    return fig


# ---------- Calculate SHAP Values ----------
def calculate_shap_values(model, X, model_type='tree'):
    """
    Calculates SHAP values for a trained model.

    Parameters:
    - model: Trained model
    - X: Feature data
    - model_type: 'tree', 'linear', or 'kernel'

    Returns:
    - explainer: SHAP explainer object
    - shap_values: SHAP values
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X)
    else:
        # Kernel explainer (slower but works for any model)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)

    return explainer, shap_values
