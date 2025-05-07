import pytest
from pyspark.sql import SparkSession, Row, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from unittest.mock import patch, MagicMock
import pandas as pd

from traffic_accident_impact.processing_spark import Preprocessing, upload_to_s3

@pytest.fixture
def sample_spark_df(spark_session: SparkSession):
    """Creates a sample Spark DataFrame for testing."""
    data = [
        Row(ID="1", Name="Alice", Age=30, Start_Time=pd.to_datetime("2023-01-15 10:00:00"), End_Time=pd.to_datetime("2023-01-15 11:00:00"), Value1=10.0, Value2=None),
        Row(ID="2", Name="Bob", Age=25, Start_Time=pd.to_datetime("2023-01-15 12:00:00"), End_Time=pd.to_datetime("2023-01-15 12:30:00"), Value1=None, Value2=20.5),
        Row(ID="3", Name="Charlie", Age=35, Start_Time=pd.to_datetime("2023-02-10 08:00:00"), End_Time=pd.to_datetime("2023-02-10 09:30:00"), Value1=15.0, Value2=25.0),
        Row(ID="4", Name="David", Age=None, Start_Time=pd.to_datetime("2023-02-10 14:00:00"), End_Time=pd.to_datetime("2023-02-10 15:00:00"), Value1=12.0, Value2=18.0),
        Row(ID="5", Name="Eve", Age=28, Start_Time=pd.to_datetime("2023-03-05 09:00:00"), End_Time=pd.to_datetime("2023-03-05 09:45:00"), Value1=100.0, Value2=-10.0) # Outlier for some tests
    ]
    schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Start_Time", TimestampType(), True),
        StructField("End_Time", TimestampType(), True),
        StructField("Value1", DoubleType(), True),
        StructField("Value2", DoubleType(), True),
    ])
    return spark_session.createDataFrame(data, schema)

@pytest.fixture
def preprocessor_instance(sample_spark_df):
    """Fixture to get an instance of the Preprocessing class."""
    return Preprocessing(sample_spark_df.copy(), outlier_column_list=[]) 

@pytest.fixture
def preprocessor_for_outliers(spark_session: SparkSession):
    """Fixture for Preprocessing instance with specific data for outlier tests."""
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("distance_mi", DoubleType(), True),
        StructField("temperature_f", DoubleType(), True),
        StructField("pressure_in", DoubleType(), True)
    ])
    data = [
        Row(id="1", distance_mi=1.0, temperature_f=50.0, pressure_in=29.0), # Normal
        Row(id="2", distance_mi=0.5, temperature_f=55.0, pressure_in=29.1), # Normal
        Row(id="3", distance_mi=100.0, temperature_f=60.0, pressure_in=29.2), # Outlier distance_mi
        Row(id="4", distance_mi=1.2, temperature_f=1000.0, pressure_in=29.3), # Outlier temperature_f
        Row(id="5", distance_mi=0.8, temperature_f=52.0, pressure_in=29.05),
        Row(id="6", distance_mi=1.1, temperature_f=58.0, pressure_in=29.15),
        Row(id="7", distance_mi=0.7, temperature_f=400.0, pressure_in=29.25), # Outlier temperature_f
        Row(id="8", distance_mi=200.0, temperature_f=500.0, pressure_in=35.0) # Outlier all, pressure_in also an outlier
    ]
    df = spark_session.createDataFrame(data, schema)
    # For these tests, the outlier_column_list in Preprocessing will be populated
    return Preprocessing(df, outlier_column_list=['distance_mi', 'temperature_f', 'pressure_in'])

# --- Tests for Preprocessing class ---

def test_preprocessor_init_column_lowercase(spark_session: SparkSession):
    """Test that column names are converted to lowercase on init."""
    data = [Row(COL_A="val1", Col_B=1)]
    schema = StructType([
        StructField("COL_A", StringType(), True),
        StructField("Col_B", IntegerType(), True),
    ])
    df = spark_session.createDataFrame(data, schema)
    preprocessor = Preprocessing(df, [])
    assert "col_a" in preprocessor.df.columns
    assert "col_b" in preprocessor.df.columns
    assert "COL_A" not in preprocessor.df.columns
    assert "Col_B" not in preprocessor.df.columns

def test_drop_columns(preprocessor_instance: Preprocessing, sample_spark_df: SparkSession):
    initial_columns = preprocessor_instance.df.columns
    columns_to_drop = ['name', 'age'] 
    
    preprocessor_instance.drop_columns(columns_to_drop)
    
    final_columns = preprocessor_instance.df.columns
    assert 'name' not in final_columns
    assert 'age' not in final_columns
    assert 'id' in final_columns # Check a column that should remain
    assert len(final_columns) == len(initial_columns) - len(columns_to_drop)

def test_drop_columns_non_existent(preprocessor_instance: Preprocessing):
    initial_df = preprocessor_instance.df
    columns_to_drop = ['non_existent_col']
    
    preprocessor_instance.drop_columns(columns_to_drop)
    
    assert preprocessor_instance.df.columns == initial_df.columns
    assert preprocessor_instance.df.count() == initial_df.count()

def test_drop_null_values(preprocessor_instance: Preprocessing):
    preprocessor_instance.drop_null_values()
    assert preprocessor_instance.df.count() == 2
    # Check that remaining rows are those without any nulls
    remaining_ids = [row.id for row in preprocessor_instance.df.select("id").collect()]
    assert "3" in remaining_ids
    assert "5" in remaining_ids

def test_fill_columns_with_monthly_mean(spark_session: SparkSession):
    """Test filling nulls with monthly mean for a specific column."""
    schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Start_Time", TimestampType(), True),
        StructField("Precipitation", DoubleType(), True)
    ])
    data = [
        Row(ID="1", Start_Time=pd.to_datetime("2023-01-10 10:00:00"), Precipitation=1.0),
        Row(ID="2", Start_Time=pd.to_datetime("2023-01-15 12:00:00"), Precipitation=None), # Null to be filled for Jan
        Row(ID="3", Start_Time=pd.to_datetime("2023-01-20 08:00:00"), Precipitation=3.0),
        Row(ID="4", Start_Time=pd.to_datetime("2023-02-05 14:00:00"), Precipitation=4.0),
        Row(ID="5", Start_Time=pd.to_datetime("2023-02-12 09:00:00"), Precipitation=None), # Null to be filled for Feb
        Row(ID="6", Start_Time=pd.to_datetime("2023-02-18 11:00:00"), Precipitation=6.0),
        Row(ID="7", Start_Time=pd.to_datetime("2023-03-01 00:00:00"), Precipitation=None)  # Null, but only one entry for March
    ]
    df = spark_session.createDataFrame(data, schema)
    
    
    df = df.select(
        F.col("ID").alias("id"), 
        F.col("Start_Time").alias("start_time"), 
        F.col("Precipitation").alias("precipitation")
    )
    # Add a dummy end_time as the fill_columns method attempts to convert it.
    df = df.withColumn("end_time", F.col("start_time")) 

    preprocessor = Preprocessing(df, [])
    preprocessor.fill_columns(col_name='precipitation')
    
    result_df = preprocessor.df
    result_data = {row.id: row.precipitation for row in result_df.select("id", "precipitation").collect()}

    # Jan: (1.0 + 3.0) / 2 = 2.0
    assert result_data["1"] == pytest.approx(1.0)
    assert result_data["2"] == pytest.approx(2.0) # Filled with Jan mean
    assert result_data["3"] == pytest.approx(3.0)
    
    # Feb: (4.0 + 6.0) / 2 = 5.0
    assert result_data["4"] == pytest.approx(4.0)
    assert result_data["5"] == pytest.approx(5.0) # Filled with Feb mean
    assert result_data["6"] == pytest.approx(6.0)
    
    
    assert result_data["7"] is None

    # Check that YearMonth and monthly_mean columns are dropped
    assert "YearMonth" not in result_df.columns
    assert "monthly_mean" not in result_df.columns

def test_fill_columns_column_not_exist(preprocessor_instance: Preprocessing):
    with pytest.raises(ValueError, match="Column non_existent_col does not exist in DataFrame"):
        preprocessor_instance.fill_columns(col_name='non_existent_col')

def test_time_processing(spark_session: SparkSession):
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("start_time", TimestampType(), True),
        StructField("end_time", TimestampType(), True)
    ])
    data = [
        Row(id="1", start_time=pd.to_datetime("2023-01-15 10:00:00"), end_time=pd.to_datetime("2023-01-15 11:30:30")),
        Row(id="2", start_time=pd.to_datetime("2022-12-31 23:59:00"), end_time=pd.to_datetime("2023-01-01 00:01:00")),
    ]
    df = spark_session.createDataFrame(data, schema)
    preprocessor = Preprocessing(df, [])
    preprocessor.time_processing()
    
    result_df = preprocessor.df
    results = {row.id: row for row in result_df.collect()}
    
    # Row 1: Duration = 1h 30m 30s = 90.5 minutes
    assert results["1"]["accident_duration(min)"] == pytest.approx(90.5)
    assert results["1"]["start_hour"] == 10
    assert results["1"]["start_day"] == 15
    assert results["1"]["start_month"] == 1
    assert results["1"]["start_year"] == 2023
    
    # Row 2: Duration = 2 minutes
    assert results["2"]["accident_duration(min)"] == pytest.approx(2.0)
    assert results["2"]["start_hour"] == 23
    assert results["2"]["start_day"] == 31
    assert results["2"]["start_month"] == 12
    assert results["2"]["start_year"] == 2022
    
    assert "start_time" not in result_df.columns
    assert "end_time" not in result_df.columns

def test_get_whiskers(preprocessor_for_outliers: Preprocessing):
    preprocessor = preprocessor_for_outliers
    # Manually calculate expected whiskers for 'distance_mi' using the Preprocessing class's logic (1.5 IQR)
    # Data for distance_mi: [1.0, 0.5, 100.0, 1.2, 0.8, 1.1, 0.7, 200.0]
    # Sorted: [0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 100.0, 200.0]
    # ApproxQuantile in Spark might give slightly different results than exact pandas.quantile
    # For simplicity, we'll test if the structure is right and values are plausible.
    # Let's use pandas to estimate for this test logic, acknowledging Spark might differ slightly.
    # df_pd_dist = pd.Series([1.0, 0.5, 100.0, 1.2, 0.8, 1.1, 0.7, 200.0])
    # q1_pd_dist = df_pd_dist.quantile(0.25) # 0.775
    # q3_pd_dist = df_pd_dist.quantile(0.75) # 75.3
    # iqr_pd_dist = q3_pd_dist - q1_pd_dist # 74.525
    # lower_pd_dist = q1_pd_dist - 1.5 * iqr_pd_dist # 0.775 - 1.5 * 74.525 = -111.0125
    # upper_pd_dist = q3_pd_dist + 1.5 * iqr_pd_dist # 75.3 + 1.5 * 74.525 = 187.0875

    whiskers = preprocessor.get_whiskers()
    
    assert 'distance_mi' in whiskers
    assert 'temperature_f' in whiskers
    assert 'pressure_in' in whiskers
    assert len(whiskers['distance_mi']) == 2
    assert isinstance(whiskers['distance_mi'][0], float) # Lower bound
    assert isinstance(whiskers['distance_mi'][1], float) # Upper bound

    # Example: For distance_mi [0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 100.0, 200.0]
    # Spark approxQuantile([0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 100.0, 200.0], [0.25, 0.75], 0.01) might be:
    # Q1 approx 0.7, Q3 approx 1.2 (if it only considers the small values due to large gap)
    # This shows the sensitivity of approxQuantile. 
    # The important part is that the function runs and returns the expected structure.
    # A more robust test would be to mock approxQuantile if precise values are needed, 
    # or test with a distribution where approxQuantile is more stable.
    
    # Let's test with a more stable column, e.g., 'pressure_in'
    # pressure_in: [29.0, 29.1, 29.2, 29.3, 29.05, 29.15, 29.25, 35.0]
    # Sorted: [29.0, 29.05, 29.1, 29.15, 29.2, 29.25, 29.3, 35.0]
    # Pandas: Q1=29.0875, Q3=29.2625, IQR=0.175
    # Lower = 29.0875 - 1.5 * 0.175 = 28.825
    # Upper = 29.2625 + 1.5 * 0.175 = 29.525
    
    # We expect Spark's approxQuantile to be close to these for 'pressure_in'
    # For this test, we mostly care about the function not crashing and returning the dict.
    # Exact value assertion for approxQuantile is tricky without mocking it.
    lb_pressure, ub_pressure = whiskers['pressure_in']
    assert lb_pressure < ub_pressure
    # Based on pandas calculation, 35.0 should be an outlier, so ub_pressure should be < 35.0
    # And lb_pressure should be around 28.825
    # These are just sanity checks, not exact value tests due to approxQuantile
    assert ub_pressure < 30.0 # Expecting something around 29.525
    assert lb_pressure > 28.0 # Expecting something around 28.825

def test_get_whiskers_column_not_in_df(spark_session: SparkSession):
    df = spark_session.createDataFrame([Row(a=1)], StructType([StructField("a", IntegerType())]))
    preprocessor = Preprocessing(df, outlier_column_list=['b'])
    whiskers = preprocessor.get_whiskers()
    assert 'b' not in whiskers # 'b' should be skipped as it's not in df
    assert not whiskers # The whiskers dict should be empty

def test_remove_outliers(preprocessor_for_outliers: Preprocessing):
    preprocessor = preprocessor_for_outliers
    # Original count: 8 rows
    # distance_mi outliers (using pandas estimated bounds for this test logic): 100.0, 200.0
    #   lower_pd_dist = -111.0125, upper_pd_dist = 187.0875. Rows with 100.0 (id=3) is NOT an outlier by this.
    #   This highlights the challenge. Let's use the 'pressure_in' bounds we estimated:
    #   lb_pressure approx 28.825, ub_pressure approx 29.525
    #   Outliers for pressure_in: 35.0 (id=8)
    # temperature_f outliers: 1000.0 (id=4), 400.0 (id=7), 500.0 (id=8)

    # For simplicity in testing remove_outliers, let's mock get_whiskers to return fixed bounds
    # This makes the test deterministic and independent of approxQuantile variations.
    fixed_whiskers = {
        'distance_mi': [-10.0, 50.0],  # Will make 100.0 (id=3) and 200.0 (id=8) outliers
        'temperature_f': [0.0, 100.0], # Will make 1000.0 (id=4), 400.0 (id=7), 500.0 (id=8) outliers
        'pressure_in': [28.0, 30.0]    # Will make 35.0 (id=8) an outlier
    }
    
    with patch.object(preprocessor, 'get_whiskers', return_value=fixed_whiskers) as mock_get_whiskers:
        preprocessor.remove_outliers(fixed_whiskers) # Pass fixed_whiskers also to the method per its signature
        # If remove_outliers internally calls self.get_whiskers(), the mock is enough.
        # The current code for remove_outliers takes `whiskers` as an argument.

    result_df = preprocessor.df
    remaining_ids = sorted([row.id for row in result_df.select("id").collect()])
    
    # Expected to remain (original data):
    # Row(id="1", distance_mi=1.0, temperature_f=50.0, pressure_in=29.0) -> Keep
    # Row(id="2", distance_mi=0.5, temperature_f=55.0, pressure_in=29.1) -> Keep
    # Row(id="3", distance_mi=100.0, ...) -> distance_mi outlier -> Remove
    # Row(id="4", ..., temperature_f=1000.0, ...) -> temperature_f outlier -> Remove
    # Row(id="5", distance_mi=0.8, temperature_f=52.0, pressure_in=29.05) -> Keep
    # Row(id="6", distance_mi=1.1, temperature_f=58.0, pressure_in=29.15) -> Keep
    # Row(id="7", ..., temperature_f=400.0, ...) -> temperature_f outlier -> Remove
    # Row(id="8", distance_mi=200.0, temperature_f=500.0, pressure_in=35.0) -> All outlier -> Remove

    expected_remaining_ids = sorted(["1", "2", "5", "6"])
    assert remaining_ids == expected_remaining_ids
    assert result_df.count() == len(expected_remaining_ids)

def test_remove_outliers_no_outlier_columns_specified(spark_session: SparkSession):
    df = spark_session.createDataFrame([Row(id="1", val=100.0)], schema="id string, val double")
    preprocessor = Preprocessing(df.copy(), outlier_column_list=[]) # No outlier columns
    initial_count = preprocessor.df.count()
    
    # Call get_whiskers (will be empty) and remove_outliers
    whiskers = preprocessor.get_whiskers()
    preprocessor.remove_outliers(whiskers)
    
    assert preprocessor.df.count() == initial_count # No change if no outlier columns

# Placeholder for more tests for Preprocessing class
# test_fill_columns
# test_get_whiskers
# test_remove_outliers
# test_time_processing

# --- Tests for upload_to_s3 --- 
@patch('traffic_accident_impact.processing_spark.boto3.client')
def test_upload_to_s3(mock_boto_client, tmp_path):
    mock_s3_instance = MagicMock()
    mock_boto_client.return_value = mock_s3_instance
    
    local_file = tmp_path / "test_log.log"
    local_file.write_text("dummy log content")
    s3_uri = "s3://my-test-bucket/logs/preprocessing.log"
    
    upload_to_s3(str(local_file), s3_uri)
    
    mock_boto_client.assert_called_once_with("s3")
    mock_s3_instance.upload_file.assert_called_once_with(
        str(local_file), 
        "my-test-bucket", 
        "logs/preprocessing.log"
    )

# Placeholder for transform_data tests (likely more integration-focused) 