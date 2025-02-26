import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from pathlib import Path

from traffic_accident_impact.config import *


def plot_box(df: pd.DataFrame, feature_name: str, fliers: bool=True) -> None:
    fig, ax = plt.subplots(1, 1)
    box = ax.boxplot(df[feature_name], showfliers = fliers)
    ax.set_title(feature_name)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"boxplot_{feature_name}.png", dpi=500)
    plt.show()
    plt.close()


def get_feature_importance(pipeline, model_name: str, output_path_frac: Path, x_train, y_train):
    # Retrieve the preprocessor and the model from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    
    # Extract importance based on the model type
    if model_name == 'Linear':
        
        # p-values estimation via statsmodels
        X_transformed = preprocessor.transform(x_train)
        X_with_intercept = sm.add_constant(X_transformed)
        model_sm = sm.OLS(y_train, X_with_intercept).fit()
        
        importance = model_sm.params[1:]
        p_values = model_sm.pvalues[1:]
        
    else:
        importance = model.feature_importances_
        p_values = np.full_like(importance, np.nan)

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance-Coeff': importance,
        'p-value': p_values
    })

    importance_df = importance_df.sort_values(by='p-value', ascending=True)
    importance_df.to_excel(output_path_frac / "feature_importance.xlsx", index=False)



def get_residuals(y_true, pred, output_path: Path):
    residuals = y_true - pred
    dw_stat = durbin_watson(residuals)

    
    fig = plt.figure(figsize=(16,5))

    ax1 = fig.add_subplot(1, 3, 1)
    plt.hist(residuals, bins=30, edgecolor='black')
    ax1.set_xlabel('Residuals')
    ax1.set_xlabel('Frequence')
    ax1.set_title('Histogram of residuals')

    ax2 = fig.add_subplot(1, 3, 2)
    plt.scatter(pred, residuals, s=15)
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')  # Ajout d'une ligne à zéro pour la référence
    ax2.set_title('Scatter plot of residuals')

    ax3 = fig.add_subplot(1, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    ax3.set_xlabel('Theoretical quantiles')
    ax3.set_ylabel('Residuals quantiles')
    ax3.set_title('QQ-residual plot')

    plt.suptitle(f"Durbin-Watson stat:  {dw_stat:.3f}", y = 1.03, fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()

    # Durbin-Watson test    
    if 1.5 <= dw_stat <= 2.5:
        return f"dw_stat = {dw_stat:.2f}  ---->  The residuals appear random (no obvious autocorrelation)."
    elif dw_stat < 1.5:
        return f"dw_stat = {dw_stat:.2f}  ---->  The residuals exhibit positive autocorrelation."
    else:
        return f"dw_stat = {dw_stat:.2f}  ---->  The residuals exhibit negative autocorrelation."


def adjusted_r2_score(y_true, y_pred, n_features: int):
    """
    Calculate the adjusted R-squared of a model.

    Parameters
    ----------
    y_true : array-like
        true values
    y_pred : array-like
        predicted values
    n_features : int
        number of features in the model

    Returns
    -------
    float
        The adjusted R-squared of the model
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)



def remove_outliers(df: pd.DataFrame, column_name: str):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.0 * IQR
    
    df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].reset_index(drop=True)
    
    return df_cleaned


