import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY 
from typer.testing import CliRunner
import pickle 
import sklearn.linear_model 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.traffic_accident_impact.modeling.training_v1_3 import ModelTraining, app as training_app 



@pytest.fixture(autouse=True)
def mock_utils_functions():
    with patch('traffic_accident_impact.modeling.training_v1_3.adjusted_r2_score') as mock_adj_r2, \
         patch('traffic_accident_impact.modeling.training_v1_3.get_feature_importance') as mock_get_fi, \
         patch('traffic_accident_impact.modeling.training_v1_3.get_residuals') as mock_get_res, \
         patch('traffic_accident_impact.modeling.training_v1_3.root_mean_squared_error') as mock_rmse:
        mock_adj_r2.return_value = 0.8
        mock_rmse.return_value = 10.0
        mock_get_res.return_value = "dw_stat = 2.0 -> No autocorrelation" 
        yield mock_adj_r2, mock_get_fi, mock_get_res, mock_rmse

@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Creates a sample Pandas DataFrame for ModelTraining tests."""
    data = {
        'numerical_col1': np.random.rand(20),
        'numerical_col2': np.random.randint(0, 100, 20),
        'categorical_col1': np.random.choice(['A', 'B', 'C'], 20),
        'categorical_col2': np.random.choice(['X', 'Y'], 20),
        'accident_duration(min)': np.random.rand(20) * 100 + 30
    }
    return pd.DataFrame(data)

@pytest.fixture
def model_training_instance(sample_training_df: pd.DataFrame) -> ModelTraining:
    return ModelTraining(frame=sample_training_df.copy(), frac=1.0)

# --- Tests for ModelTraining class ---

def test_model_training_init(sample_training_df: pd.DataFrame):
    mt = ModelTraining(frame=sample_training_df.copy(), frac=0.5)
    assert mt.frac == 0.5
    assert len(mt.df) == int(len(sample_training_df) * 0.5)
    assert mt.df['accident_duration(min)'].equals(sample_training_df.loc[mt.df.index, 'accident_duration(min)'])


@patch('sklearn.model_selection.train_test_split')
def test_data_preparation(mock_train_test_split, model_training_instance: ModelTraining, sample_training_df: pd.DataFrame):
    mt = model_training_instance
    X_internal = sample_training_df.drop('accident_duration(min)', axis=1)
    y_internal = sample_training_df['accident_duration(min)']
    
    X_train_dummy = X_internal.sample(frac=0.8, random_state=1)
    X_test_dummy = X_internal.drop(X_train_dummy.index)
    y_train_dummy = y_internal.loc[X_train_dummy.index]
    y_test_dummy = y_internal.loc[X_test_dummy.index]
    mock_train_test_split.return_value = (X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy)

    mt.data_preparation()

    assert mt.X.equals(X_internal)
    assert mt.y.equals(y_internal)
    
    assert 'numerical_col1' in mt.preprocessor.transformers_[0][2]
    assert 'categorical_col1' in mt.preprocessor.transformers_[1][2]
    
    mock_train_test_split.assert_called_once()
    pd.testing.assert_frame_equal(mock_train_test_split.call_args[0][0], X_internal)
    pd.testing.assert_series_equal(mock_train_test_split.call_args[0][1], y_internal)
    assert mock_train_test_split.call_args[1]['test_size'] == 0.2

    assert mt.X_train.equals(X_train_dummy)
    assert mt.y_test.equals(y_test_dummy)

@patch('pickle.dump')
def test_save_training_results(mock_pickle_dump, model_training_instance: ModelTraining, tmp_path, mock_utils_functions):
    mt = model_training_instance
    mt.data_preparation()

    mock_adj_r2, mock_get_fi, mock_get_res, mock_rmse = mock_utils_functions

    mock_best_estimator = MagicMock()
    mock_best_estimator.predict.side_effect = [np.array([1]*len(mt.y_train)), np.array([1]*len(mt.y_test))] 
    
    mock_random_search_cv_results = MagicMock() # Mock for RandomizedSearchCV instance
    mock_random_search_cv_results.best_estimator_ = mock_best_estimator 

    model_name = "TestModel"
    output_path_frac = tmp_path / model_name / f"frac-{mt.frac}"
    output_path_frac.mkdir(parents=True, exist_ok=True)

    mt.save_training_results(mock_random_search_cv_results, model_name, output_path_frac)

    # Check pickle dump
    expected_pkl_path = output_path_frac / f"{model_name}-frac-{mt.frac}.pkl"
    mock_pickle_dump.assert_called_once_with(mock_random_search_cv_results, ANY) # ANY for file object
    assert mock_pickle_dump.call_args[0][1].name == str(expected_pkl_path) # Check file name from file object

    # Check predict calls
    assert mock_best_estimator.predict.call_count == 2
    # Check calls to utils functions
    mock_adj_r2.assert_any_call(mt.y_train, ANY, mt.X_train.shape[1])
    mock_adj_r2.assert_any_call(mt.y_test, ANY, mt.X_test.shape[1])
    mock_rmse.assert_any_call(mt.y_train, ANY)
    mock_rmse.assert_any_call(mt.y_test, ANY)
    mock_get_fi.assert_called_once_with(mock_best_estimator, model_name, output_path_frac, mt.X_train, mt.y_train)
    mock_get_res.assert_any_call(mt.y_train, ANY, output_path_frac / 'train_residuals.png')
    mock_get_res.assert_any_call(mt.y_test, ANY, output_path_frac / 'test_residuals.png')

@pytest.mark.parametrize(
    "model_name_cli, expected_model_type, expected_param_keys, n_iter_val, nb_cv_val",
    [
        ("linear", sklearn.linear_model.LinearRegression, [], 20, 3), # Default params for linear
        ("randomforest", RandomForestRegressor, ["model__n_estimators", "model__max_depth"], 20, 3),
        ("xgboost", XGBRegressor, ["model__n_estimators", "model__learning_rate"], 20, 3)
    ]
)
@patch('sklearn.model_selection.RandomizedSearchCV')
@patch('sklearn.pipeline.Pipeline')
def test_randomized_searchCV_parametrized(
    MockPipeline, MockRandomizedSearchCV, 
    model_name_cli, expected_model_type, expected_param_keys, n_iter_val, nb_cv_val, # parametrize args
    model_training_instance: ModelTraining, tmp_path, mock_utils_functions
):
    mt = model_training_instance
    mt.data_preparation()

    mock_rscv_instance = MagicMock()
    mock_rscv_instance.best_estimator_ = MagicMock()
    mock_rscv_instance.best_estimator_.predict.return_value = np.array([1]*len(mt.y_train))
    MockRandomizedSearchCV.return_value = mock_rscv_instance

    mock_pipeline_instance = MagicMock()
    MockPipeline.return_value = mock_pipeline_instance

    mt.randomized_searchCV(model_name=model_name_cli, n_iter=n_iter_val, nb_cv=nb_cv_val)

    MockPipeline.assert_called_once()
    pipeline_steps = MockPipeline.call_args[1]['steps']
    assert pipeline_steps[0][0] == 'preprocessor'
    assert pipeline_steps[0][1] == mt.preprocessor
    assert isinstance(pipeline_steps[1][1], expected_model_type)
    assert pipeline_steps[1][0] == 'model'

    MockRandomizedSearchCV.assert_called_once()
    rscv_args = MockRandomizedSearchCV.call_args
    assert rscv_args[1]['estimator'] == mock_pipeline_instance
    assert rscv_args[1]['n_iter'] == n_iter_val
    assert rscv_args[1]['cv'] == nb_cv_val
    assert rscv_args[1]['scoring'] == 'neg_root_mean_squared_error'
    actual_param_distributions = rscv_args[1]['param_distributions']
    if not expected_param_keys: # For Linear regression, params are empty
        assert actual_param_distributions == {}
    else:
        assert all(key in actual_param_distributions for key in expected_param_keys)
        assert len(actual_param_distributions) >= len(expected_param_keys)

    mock_rscv_instance.fit.assert_called_once_with(mt.X_train, mt.y_train)
    
    # Check that save_training_results was called (verified via mock_get_fi)
    _, mock_get_fi, _, _ = mock_utils_functions
    # Check parent output path to be robust against frac or casing variations in full path.
    # Assumes path uses lowercase model name.
    expected_output_parent_path = tmp_path / model_name_cli.lower() 
    
    mock_get_fi.assert_called_with(mock_rscv_instance.best_estimator_, model_name_cli.lower(), ANY, mt.X_train, mt.y_train)
    assert mock_get_fi.call_args[0][2].parent == expected_output_parent_path

# --- Test for train_save (CLI command) ---

@patch('pandas.read_parquet')
@patch('traffic_accident_impact.modeling.training_v1_3.ModelTraining')
@patch('loguru.logger.add')
def test_train_save_cli(mock_logger_add, MockModelTraining, mock_pd_read_parquet, tmp_path):
    runner = CliRunner()

    mock_input_df = pd.DataFrame({'feature': [1,2,3], 'accident_duration(min)': [10,20,30]})
    mock_pd_read_parquet.return_value = mock_input_df

    mock_mt_instance = MagicMock()
    MockModelTraining.return_value = mock_mt_instance

    input_file = tmp_path / "final_data.parquet"
    pd.DataFrame().to_parquet(input_file)

    model_name_cli = "linear"
    frac_cli = 0.8

    result = runner.invoke(training_app, [
        "--parquet-file-path", str(input_file),
        "--frac", str(frac_cli),
        "--model-name", model_name_cli
    ])

    assert result.exit_code == 0, f"CLI command failed: {result.stdout}\n{result.stderr}"

    mock_pd_read_parquet.assert_called_once_with(input_file)
    MockModelTraining.assert_called_once_with(frame=mock_input_df, frac=frac_cli)

    mock_mt_instance.data_preparation.assert_called_once()
    mock_mt_instance.randomized_searchCV.assert_called_once_with(model_name=model_name_cli.lower(), n_iter=20, nb_cv=3)
    mock_logger_add.assert_called()
