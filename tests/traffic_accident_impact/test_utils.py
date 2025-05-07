import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Supposons que votre projet est structuré de manière à ce que vous puissiez importer utils comme ceci:
# Si ce n'est pas le cas, ajustez le chemin d'importation.
# Il est courant d'avoir un __init__.py dans le dossier tests et d'ajouter src au PYTHONPATH
# ou d'installer le projet en mode éditable (pdm install -e .)
from traffic_accident_impact import utils # Ajustez si nécessaire, ex: from src.traffic_accident_impact import utils

# --- Tests for remove_outliers ---

def test_remove_outliers_no_outliers():
    data = {'value': [10, 12, 11, 13, 10, 14, 12, 11, 13, 15]}
    df = pd.DataFrame(data)
    # Q1 = 10.75, Q3 = 13.0, IQR = 2.25
    # Lower bound = 10.75 - 1.5 * 2.25 = 7.375
    # Upper bound = 13.0 + 1.0 * 2.25 = 15.25
    # No outliers expected with these bounds
    cleaned_df = utils.remove_outliers(df.copy(), 'value')
    assert cleaned_df.shape[0] == df.shape[0], "Should not remove rows if no outliers according to the 1.0*IQR upper rule"

def test_remove_outliers_with_upper_outliers():
    # Note: utils.remove_outliers uses Q3 + 1.0 * IQR for upper bound
    data = {'value': [10, 12, 11, 13, 10, 14, 12, 11, 13, 20, 22]} # 20, 22 are outliers
    df = pd.DataFrame(data)
    # Q1 = 11.0, Q3 = 13.5, IQR = 2.5
    # Lower bound = 11.0 - 1.5 * 2.5 = 7.25
    # Upper bound = 13.5 + 1.0 * 2.5 = 16.0
    # Expected to keep values <= 16.0
    cleaned_df = utils.remove_outliers(df.copy(), 'value')
    assert cleaned_df.shape[0] == 9
    assert 20 not in cleaned_df['value'].values
    assert 22 not in cleaned_df['value'].values

def test_remove_outliers_with_lower_outliers():
    data = {'value': [1, 2, 10, 12, 11, 13, 10, 14, 12, 11, 13]} # 1, 2 are outliers
    df = pd.DataFrame(data)
    # Q1 = 10.0, Q3 = 12.5, IQR = 2.5
    # Lower bound = 10.0 - 1.5 * 2.5 = 6.25
    # Upper bound = 12.5 + 1.0 * 2.5 = 15.0
    # Expected to keep values >= 6.25
    cleaned_df = utils.remove_outliers(df.copy(), 'value')
    assert cleaned_df.shape[0] == 9
    assert 1 not in cleaned_df['value'].values
    assert 2 not in cleaned_df['value'].values

def test_remove_outliers_empty_df():
    df = pd.DataFrame({'value': []})
    cleaned_df = utils.remove_outliers(df.copy(), 'value')
    assert cleaned_df.empty

def test_remove_outliers_single_value_df():
    df = pd.DataFrame({'value': [10]})
    cleaned_df = utils.remove_outliers(df.copy(), 'value')
    assert cleaned_df.shape[0] == 1 # Should keep the single value

# --- Tests for adjusted_r2_score ---

def test_adjusted_r2_score_perfect_fit():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    n_features = 1
    # R2 should be 1, so adjusted R2 should also be 1
    adj_r2 = utils.adjusted_r2_score(y_true, y_pred, n_features)
    assert pytest.approx(adj_r2) == 1.0

def test_adjusted_r2_score_no_fit():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1]) # Example of a poor fit
    n_features = 1
    # R2 will be < 0 for this case.
    # Let's calculate R2 manually:
    # SS_tot = sum((y_true - y_true.mean())^2) = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) = 4+1+0+1+4 = 10
    # SS_res = sum((y_true - y_pred)^2) = ((1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2) = 16+4+0+4+16 = 40
    # R2 = 1 - SS_res / SS_tot = 1 - 40/10 = 1 - 4 = -3.0
    # n = 5, p = 1
    # Adj R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    # Adj R2 = 1 - (1 - (-3.0)) * (5 - 1) / (5 - 1 - 1)
    # Adj R2 = 1 - (4) * 4 / 3 = 1 - 16/3 = 1 - 5.333... = -4.333...
    adj_r2 = utils.adjusted_r2_score(y_true, y_pred, n_features)
    assert pytest.approx(adj_r2) == 1 - (1 - (-3.0)) * (5 - 1) / (5 - 1 - 1)

def test_adjusted_r2_score_less_features():
    # R2 = 0.5, n=10, p1=2
    # adj_r2_1 = 1 - (1-0.5)*(10-1)/(10-2-1) = 1 - 0.5 * 9 / 7 = 1 - 4.5/7 = 1 - 0.6428 = 0.3571
    # R2 = 0.5, n=10, p2=5
    # adj_r2_2 = 1 - (1-0.5)*(10-1)/(10-5-1) = 1 - 0.5 * 9 / 4 = 1 - 4.5/4 = 1 - 1.125 = -0.125
    # For the same R2, more features should result in a lower or more penalized adjusted R2
    
    # We need to construct y_true, y_pred that give R2 = 0.5
    # Let y_true.mean() = 0. SS_tot = sum(y_true^2)
    # R2 = 1 - SS_res / SS_tot => 0.5 = 1 - SS_res / SS_tot => SS_res / SS_tot = 0.5 => SS_res = 0.5 * SS_tot
    y_true = np.array([-2, -1, 0, 1, 2, -2, -1, 0, 1, 2]) # mean=0, SS_tot = 4+1+0+1+4+4+1+0+1+4 = 20
    # We need SS_res = 10
    # Example: y_pred where sum((y_true-y_pred)^2) = 10
    y_pred = y_true - np.array([1,1,1,1,1, -1,-1,-1,-1,-1]) # sum of squares of residuals = 1*5 + (-1)^2*5 = 10
                                                          # ([-3,-2,-1,0,1, -1,0,1,2,3]) - Not this one
                                                          # residuals = [-1,-1,-1,-1,-1, 1,1,1,1,1], sum(res^2)=10
    # Let's verify R2 with sklearn directly
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, np.array([-1,0,1,2,1, -3,-2,-1,0,1])) # Found a y_pred that gives ~0.5 R2
    # For simplicity, let's use a known R2 value and check adjustment.
    # If r2 = 0.5, n=10, n_features=2
    # adj_r2 = 1 - (1 - 0.5) * (10 - 1) / (10 - 2 - 1) = 1 - 0.5 * 9 / 7 = 1 - 4.5/7 approx 0.3571
    
    # Mock r2_score to control R2 value directly for testing the adjustment logic
    with patch('traffic_accident_impact.utils.r2_score') as mock_r2:
        mock_r2.return_value = 0.5
        adj_r2_f2 = utils.adjusted_r2_score(y_true, y_pred, n_features=2) # n=10
        assert pytest.approx(adj_r2_f2) == 1 - (1 - 0.5) * (10 - 1) / (10 - 2 - 1)

        adj_r2_f5 = utils.adjusted_r2_score(y_true, y_pred, n_features=5) # n=10
        assert pytest.approx(adj_r2_f5) == 1 - (1 - 0.5) * (10 - 1) / (10 - 5 - 1)
        assert adj_r2_f5 < adj_r2_f2

# --- Tests for plot_box ---

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show') # Mock show to prevent GUI pop-ups during tests
@patch('matplotlib.pyplot.close')
def test_plot_box(mock_close, mock_show, mock_savefig, tmp_path):
    """Test that plot_box runs and attempts to save a figure."""
    data = {'feature1': [1, 2, 3, 4, 5, 100]} # 100 is an outlier
    df = pd.DataFrame(data)
    feature_name = 'feature1'
    output_file = tmp_path / "boxplot.png"

    utils.plot_box(df, feature_name, output_file, fliers=True)

    mock_savefig.assert_called_once_with(output_file, bbox_inches='tight', dpi=300)
    mock_show.assert_called_once()
    mock_close.assert_called_once()

# --- Tests for get_feature_importance ---

@patch('pandas.DataFrame.to_excel') # Mock the Excel writing part
@patch('statsmodels.api.OLS') # Mock OLS for linear model path
def test_get_feature_importance_linear_model(mock_ols, mock_to_excel, tmp_path):
    # Mocking the pipeline and its components
    mock_model = MagicMock()
    # mock_model.coef_ = np.array([0.5, -0.3]) # Not directly used, params come from OLS

    mock_preprocessor = MagicMock()
    mock_preprocessor.get_feature_names_out.return_value = ['feature_A', 'feature_B']
    mock_preprocessor.transform.return_value = np.array([[1,2],[3,4],[5,6]]) # Transformed X data

    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'preprocessor': mock_preprocessor, 'model': mock_model}

    # Mocking OLS results
    mock_ols_fit_results = MagicMock()
    mock_ols_fit_results.params = np.array([0.1, 0.5, -0.3]) # Intercept, coef_A, coef_B
    mock_ols_fit_results.pvalues = np.array([0.5, 0.01, 0.04]) # Intercept_pval, pval_A, pval_B
    mock_ols.return_value.fit.return_value = mock_ols_fit_results
    
    x_train_dummy = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    y_train_dummy = pd.Series([10,11,12])
    output_dir = tmp_path

    utils.get_feature_importance(mock_pipeline, 'Linear', output_dir, x_train_dummy, y_train_dummy)

    mock_preprocessor.transform.assert_called_once_with(x_train_dummy)
    # Check that OLS was called with data including an intercept (constant)
    assert mock_ols.call_args[0][1].shape[1] == mock_preprocessor.transform.return_value.shape[1] + 1 
    
    # Check what was passed to to_excel
    assert mock_to_excel.call_count == 1
    call_args, _ = mock_to_excel.call_args
    excel_path = call_args[0]
    df_importance = mock_to_excel.call_args.args[1] # Accessing via .args is more robust if it's a positional arg
    if isinstance(df_importance, pd.DataFrame):
        df_output = df_importance
    else: # if it's a keyword argument 'df' or similar in a real scenario
        df_output = mock_to_excel.call_args.kwargs['df'] 
        # In this mocked scenario, pandas.DataFrame.to_excel is called on an instance,
        # so the first argument to the mocked method is the DataFrame itself.
        # So the path is args[0] and the DataFrame is the instance it's called on.
        # This means my initial mock_to_excel.call_args.args[1] was wrong.
        # The DataFrame is the `self` in `df.to_excel(path)`
        # For a direct mock of `pd.DataFrame.to_excel`, the args list [path, **kwargs]
        # Let's assume to_excel is called on an instance. df_output is that instance.
        # No, the mock is on `pd.DataFrame.to_excel` itself. So the instance is the first arg.
        # It seems the way I'm trying to get df_importance is tricky with instance method mocks.
        # The easiest is to check the path and if the columns are what we expect.

    assert excel_path == output_dir / "feature_importance.xlsx"
    # Minimal check on DataFrame content based on known feature names and expected columns
    # This part can be tricky if the DataFrame is not directly passed as an argument to the mock
    # For instance method mocks, the instance itself is `call_args.args[0]` if it's `instance.method(arg1, arg2)`
    # but to_excel is `df.to_excel(path, index=...)`. So path is arg0 for the method.
    # The DataFrame instance is available in the mock object if needed, but let's check call_args.
    # The DataFrame being saved is actually the `self` of the `to_excel` method.
    # To correctly get the DataFrame passed to `to_excel`:
    # We would need to inspect the mock object itself or how it was called.
    # A simpler check for now:
    saved_df_arg = mock_to_excel.call_args.args[0] # This is the DataFrame instance
    assert isinstance(saved_df_arg, pd.DataFrame)
    assert list(saved_df_arg.columns) == ['Feature', 'Importance-Coeff', 'p-value']
    assert len(saved_df_arg) == 2 # Two features
    assert 'feature_A' in saved_df_arg['Feature'].values
    assert pytest.approx(saved_df_arg[saved_df_arg['Feature'] == 'feature_A']['Importance-Coeff'].iloc[0]) == 0.5
    assert pytest.approx(saved_df_arg[saved_df_arg['Feature'] == 'feature_A']['p-value'].iloc[0]) == 0.01


@patch('pandas.DataFrame.to_excel')
def test_get_feature_importance_tree_model(mock_to_excel, tmp_path):
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.7, 0.3])

    mock_preprocessor = MagicMock()
    mock_preprocessor.get_feature_names_out.return_value = ['feature_X', 'feature_Y']

    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'preprocessor': mock_preprocessor, 'model': mock_model}

    x_train_dummy = pd.DataFrame({'X': [1,2,3], 'Y': [4,5,6]})
    y_train_dummy = pd.Series([10,11,12]) # Not used by this path directly but required by func signature
    output_dir = tmp_path

    utils.get_feature_importance(mock_pipeline, 'RandomForest', output_dir, x_train_dummy, y_train_dummy)
    
    mock_to_excel.assert_called_once()
    saved_df_arg = mock_to_excel.call_args.args[0] # DataFrame instance
    excel_path_arg = mock_to_excel.call_args.args[1] # Path argument to to_excel

    assert excel_path_arg == output_dir / "feature_importance.xlsx"
    assert isinstance(saved_df_arg, pd.DataFrame)
    assert list(saved_df_arg.columns) == ['Feature', 'Importance-Coeff', 'p-value']
    assert len(saved_df_arg) == 2
    assert 'feature_X' in saved_df_arg['Feature'].values
    assert saved_df_arg[saved_df_arg['Feature'] == 'feature_X']['Importance-Coeff'].iloc[0] == 0.7
    assert pd.isna(saved_df_arg[saved_df_arg['Feature'] == 'feature_X']['p-value'].iloc[0]) # p-values are NaN for tree models

# --- Tests for get_residuals ---

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
@patch('statsmodels.stats.stattools.durbin_watson')
@patch('scipy.stats.probplot') # probplot is called as stats.probplot
def test_get_residuals(mock_probplot, mock_durbin_watson, mock_close, mock_show, mock_savefig, tmp_path):
    y_true = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pred = pd.Series(  [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9])
    output_file = tmp_path / "residuals.png"

    # Test case 1: No autocorrelation
    mock_durbin_watson.return_value = 2.0
    result_str = utils.get_residuals(y_true, pred, output_file)
    mock_savefig.assert_called_with(output_file, dpi=500)
    mock_probplot.assert_called()
    assert "dw_stat = 2.00" in result_str
    assert "no obvious autocorrelation" in result_str

    # Test case 2: Positive autocorrelation
    mock_durbin_watson.return_value = 1.0
    result_str_pos = utils.get_residuals(y_true, pred, output_file)
    assert "dw_stat = 1.00" in result_str_pos
    assert "positive autocorrelation" in result_str_pos

    # Test case 3: Negative autocorrelation
    mock_durbin_watson.return_value = 3.0
    result_str_neg = utils.get_residuals(y_true, pred, output_file)
    assert "dw_stat = 3.00" in result_str_neg
    assert "negative autocorrelation" in result_str_neg

    assert mock_savefig.call_count == 3
    assert mock_show.call_count == 0 # Original function does not call plt.show() for get_residuals
    assert mock_close.call_count == 3
    assert mock_probplot.call_count == 3

# --- Placeholder for other utils tests ---
# Will add tests for plot_box, get_feature_importance, get_residuals later 