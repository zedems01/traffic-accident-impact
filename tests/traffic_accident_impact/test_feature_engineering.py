import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.traffic_accident_impact.feature_engineering import FeatureEngineering, encoding, app as feature_engineering_app


@pytest.fixture
def sample_fe_df() -> pd.DataFrame:
    """Creates a sample Pandas DataFrame for feature engineering tests."""
    data = {
        'weather_timestamp': [pd.Timestamp('2023-01-01 10:00:00')] * 6,
        'airport_code': ['KJFK'] * 6,
        'country': ['US'] * 6,
        'source': ['MapQuest'] * 6,
        'turning_loop': [False] * 6,
        'street': ['Main St'] * 6,
        'city': ['Anytown'] * 6,
        'county': ['Anycounty'] * 6,
        'zipcode': ['12345'] * 6,
        'amenity': [False] * 6, 'bump': [False] * 6, 'crossing': [False] * 6, 'give_way': [False] * 6,
        'junction': [False] * 6, 'no_exit': [False] * 6, 'railway': [False] * 6, 'roundabout': [False] * 6,
        'station': [False] * 6, 'stop': [False] * 6, 'traffic_calming': [False] * 6, 'traffic_signal': [False] * 6,
        'sunrise_sunset': ['Day'] * 6, 'civil_twilight': ['Day'] * 6, 'nautical_twilight': ['Day'] * 6, 'astronomical_twilight': ['Day'] * 6,
        'accident_duration(min)': [30, 45, 60, 25, 50, 1000], # 1000 is an outlier
        'start_lng': [-73.0, -74.0, -73.5, 0.0, 10.0, -10.0],
        'start_hour': [8, 12, 16, 0, 23, 10],
        'start_day': [1, 2, 3, 4, 5, 6],
        'start_month': [1, 3, 6, 9, 12, 7],
        'wind_direction': ['N', 'NE', 'East', 'SouthWest', 'calm', 'Variable'],
        'state': ['NY', 'CA', 'TX', 'FL', 'ME', 'UNKNOWN_STATE'],
        'weather_condition': ['Clear', 'Rain', 'Mostly Cloudy', 'Fog', 'Snow', 'UNKNOWN_WEATHER']
    }
    return pd.DataFrame(data)

@pytest.fixture
@patch('traffic_accident_impact.feature_engineering.remove_outliers')
def feature_engineering_instance(mock_remove_outliers, sample_fe_df: pd.DataFrame) -> FeatureEngineering:
    """Fixture to get an instance of FeatureEngineering, mocking remove_outliers."""
    mock_remove_outliers.side_effect = lambda df, col: df 
    return FeatureEngineering(frame=sample_fe_df.copy())

# --- Test for 'encoding' function ---
def test_encoding_function():
    df = pd.DataFrame({'hour': [0, 6, 12, 18]})
    encoding(df, 'hour', 24)
    assert 'sin_hour' in df.columns
    assert 'cos_hour' in df.columns
    assert pytest.approx(df['sin_hour'].iloc[0]) == 0.0 # sin(0)
    assert pytest.approx(df['cos_hour'].iloc[0]) == 1.0 # cos(0)
    assert pytest.approx(df['sin_hour'].iloc[1]) == 1.0 # sin(2*pi*6/24) = sin(pi/2)
    assert pytest.approx(df['cos_hour'].iloc[1]) == 0.0 # cos(pi/2)
    assert pytest.approx(df['sin_hour'].iloc[2]) == 0.0 # sin(pi)
    assert pytest.approx(df['cos_hour'].iloc[2]) == -1.0 # cos(pi)

# --- Tests for FeatureEngineering class ---

def test_feature_engineering_init(sample_fe_df: pd.DataFrame):
    """Test initialization, column dropping, and call to remove_outliers."""
    with patch('traffic_accident_impact.feature_engineering.remove_outliers') as mock_ro:
        mock_ro.side_effect = lambda df, col: df # Ensure it returns a DataFrame
        fe = FeatureEngineering(frame=sample_fe_df.copy())
        
        # Check columns_to_drop defined in feature_engineering.py were dropped
        assert 'airport_code' not in fe.df.columns
        assert 'street' not in fe.df.columns
        assert 'amenity' not in fe.df.columns
        assert 'sunrise_sunset' not in fe.df.columns
        
        # Check that 'accident_duration(min)'is still there
        assert 'accident_duration(min)' in fe.df.columns
        
        # Check that remove_outliers was called
        mock_ro.assert_called_once_with(fe.df, 'accident_duration(min)')

def test_cyclic_encoding(feature_engineering_instance: FeatureEngineering):
    fe = feature_engineering_instance
    fe.cyclic_encoding()
    
    # Check lng
    assert 'sin_start_lng' in fe.df.columns and 'cos_start_lng' in fe.df.columns
    assert 'start_lng' not in fe.df.columns
    # Check hour
    assert 'sin_start_hour' in fe.df.columns and 'cos_start_hour' in fe.df.columns
    assert 'start_hour' not in fe.df.columns
    # Check day (period 7)
    assert 'sin_start_day' in fe.df.columns and 'cos_start_day' in fe.df.columns
    assert 'start_day' not in fe.df.columns
    # Check month
    assert 'sin_start_month' in fe.df.columns and 'cos_start_month' in fe.df.columns
    assert 'start_month' not in fe.df.columns
    # Check wind_direction -> wind_angle -> sin/cos
    assert 'sin_wind_angle' in fe.df.columns and 'cos_wind_angle' in fe.df.columns
    assert 'wind_direction' not in fe.df.columns
    assert 'wind_angle' not in fe.df.columns
    
    # Check specific value for wind_direction 'N' mapped to 0 degrees
    # Original df had 'N' at index 0 for wind_direction
    # Sample_fe_df has 'N' at index 0. wind_to_angle['n']=0. sin(0)=0, cos(0)=1
    assert pytest.approx(fe.df['sin_wind_angle'].iloc[0]) == 0.0
    assert pytest.approx(fe.df['cos_wind_angle'].iloc[0]) == 1.0
    # 'calm' also mapped to 0
    assert pytest.approx(fe.df['sin_wind_angle'].iloc[4]) == 0.0
    assert pytest.approx(fe.df['cos_wind_angle'].iloc[4]) == 1.0
    # 'Variable' mapped to 180 deg. sin(pi)=0, cos(pi)=-1
    assert pytest.approx(fe.df['sin_wind_angle'].iloc[5]) == 0.0
    assert pytest.approx(fe.df['cos_wind_angle'].iloc[5]) == -1.0

def test_grouping_states(feature_engineering_instance: FeatureEngineering):
    fe = feature_engineering_instance
    fe.grouping_states()
    
    assert 'state_group' in fe.df.columns
    assert 'state' not in fe.df.columns
    
    # 'NY' is in urban_states
    assert fe.df['state_group'].iloc[0] == 'urban'
    # 'ME' is in rural_states
    assert fe.df['state_group'].iloc[4] == 'rural'
    # 'UNKNOWN_STATE' should be 'unknown'
    assert fe.df['state_group'].iloc[5] == 'unknown'

def test_grouping_weathers(feature_engineering_instance: FeatureEngineering):
    fe = feature_engineering_instance
    fe.grouping_weathers()
    
    assert 'weather_group' in fe.df.columns
    assert 'weather_condition' not in fe.df.columns
    
    # 'Clear' -> 'clear'
    assert fe.df['weather_group'].iloc[0] == 'clear'
    # 'Rain' -> 'precipitation'
    assert fe.df['weather_group'].iloc[1] == 'precipitation'
    # 'Fog' -> 'obscured'
    assert fe.df['weather_group'].iloc[3] == 'obscured'
    # 'UNKNOWN_WEATHER' -> 'unknown' (due to fillna)
    assert fe.df['weather_group'].iloc[5] == 'unknown'

# --- Test for transform_save (CLI command) ---

@patch('pandas.read_parquet')
@patch('traffic_accident_impact.feature_engineering.FeatureEngineering')
@patch('pandas.DataFrame.to_parquet')
@patch('loguru.logger.add')
def test_transform_save_cli(mock_logger_add, mock_df_to_parquet, MockFeatureEngineering, mock_pd_read_parquet, tmp_path):
    runner = CliRunner()
    
    mock_input_df = pd.DataFrame({'some_col': [1, 2]})
    mock_pd_read_parquet.return_value = mock_input_df
    
    mock_fe_instance = MagicMock()
    MockFeatureEngineering.return_value = mock_fe_instance
    # Simulate the final df on the instance
    final_df_to_save = pd.DataFrame({'engineered': ['a', 'b'], 'shape_col': [0,0]})
    mock_fe_instance.df = final_df_to_save
    # Mock shape attribute for logging
    mock_fe_instance.df.shape = (len(final_df_to_save), len(final_df_to_save.columns))
    mock_fe_instance.df.columns = final_df_to_save.columns

    input_file = tmp_path / "input_fe.parquet"
    output_file = tmp_path / "output_fe.parquet"
    pd.DataFrame().to_parquet(input_file) # Dummy file

    result = runner.invoke(feature_engineering_app, [
        "--data-path", str(input_file),
        "--output-path", str(output_file)
    ])
    
    assert result.exit_code == 0, f"CLI command failed: {result.stdout}"
    
    mock_pd_read_parquet.assert_called_once_with(input_file)
    MockFeatureEngineering.assert_called_once_with(mock_input_df)
    
    mock_fe_instance.cyclic_encoding.assert_called_once()
    mock_fe_instance.grouping_states.assert_called_once()
    mock_fe_instance.grouping_weathers.assert_called_once()
    
    # Check save
    saved_df_instance = mock_df_to_parquet.call_args[0][0]
    output_path_arg = mock_df_to_parquet.call_args[0][1]

    assert output_path_arg == output_file
    pd.testing.assert_frame_equal(saved_df_instance, final_df_to_save)
    mock_logger_add.assert_called() 