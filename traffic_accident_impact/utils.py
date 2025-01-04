from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import root_mean_squared_error, r2_score
from colorama import Fore, Style, init

from traffic_accident_impact.config import *

init(autoreset=True) 

def plot_box(df: pd.DataFrame, feature_name: str, fliers: bool=True) -> None:
    fig, ax = plt.subplots(1, 1)
    box = ax.boxplot(df[feature_name], showfliers = fliers)
    ax.set_title(feature_name)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"boxplot_{feature_name}.png", dpi=500)
    plt.show()
    plt.close()


def plot_residuals(y_true, pred, title):
    """
    Plot the residuals of a model.

    Parameters
    ----------
    y_true : array-like
        true values
    pred : array-like
        predicted values
    title : str
        title of the plot

    Returns
    -------
    None
    """
    residus = y_true - pred
    
    fig = plt.figure(figsize=(16,5))

    ax1 = fig.add_subplot(1, 3, 1)
    plt.hist(residus, bins=30, edgecolor='black')
    ax1.set_xlabel('Residuals')
    ax1.set_xlabel('Frequence')
    ax1.set_title('Histogram of residuals')

    ax2 = fig.add_subplot(1, 3, 2)
    plt.scatter(pred, residus, s=15)
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')  # Ajout d'une ligne à zéro pour la référence
    ax2.set_title('Scatter plot of residuals')

    ax3 = fig.add_subplot(1, 3, 3)
    stats.probplot(residus, dist="norm", plot=plt)
    ax3.set_xlabel('Theoretical quantiles')
    ax3.set_ylabel('Residuals quantiles')
    ax3.set_title('QQ-residual plot')

    plt.suptitle(title, y = 1.03, fontsize=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{title}.png", dpi=500)
    plt.show()
    plt.close()


def adjusted_r2_score(y_true, y_pred, n_features):
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


def format_text(text:str, color: str="green", style: str="bold"):
    """
    Format text with a color and a style using colorama.
    
    Args:
    -----
        text (str): The text to format.
        color (str): The color of the text. Options: "red", "green", "yellow", "blue", "magenta", "cyan", "white", "default".
        style (str): The style of the text. Options: "bold", "underline", "default".
    
    Returns:
    -------
        str: Formatted text.
    """
    colors = {
        "default": "",
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
    }
    
    styles = {
        "default": "",
        "bold": Style.BRIGHT,
        "underline": "\033[4m",
    }
    
    chosen_color = colors.get(color.lower(), colors["default"])
    chosen_style = styles.get(style.lower(), styles["default"])
    
    return f"{chosen_style}{chosen_color}{text}{Style.RESET_ALL}"



