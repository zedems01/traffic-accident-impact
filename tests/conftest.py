import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("PySparkTest") \
        .getOrCreate()
    yield spark
    spark.stop() 