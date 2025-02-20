from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, mean as spark_mean, when, avg, hour, dayofmonth, month, year, unix_timestamp
from pyspark.sql.window import Window
import argparse
import boto3
from urllib.parse import urlparse
from loguru import logger


class Preprocessing:
    def __init__(self, df: DataFrame, outlier_column_list: list[str]) -> None:
        if not isinstance(df, DataFrame):
            raise ValueError("df must be a PySpark DataFrame")
        
        self.df = df.select([col(c).alias(c.lower()) for c in self.df.columns])
        self.outlier_column_list = outlier_column_list if outlier_column_list else []
        
    def drop_columns(self, column_list: list[str]) -> None:
        try:
            self.df = self.df.drop(*column_list)
            logger.success(f"Dropped columns: {column_list}.....Lines: {self.df.count()}, Columns: {len(self.df.columns)}")
        except Exception as e:
            logger.error(f"Error dropping columns {column_list}: {e}")

    def drop_null_values(self) -> None:
        try:
            self.df = self.df.na.drop()
            logger.success(f"Dropped rows with null values.....Lines: {self.df.count()}, Columns: {len(self.df.columns)}")
        except Exception as e:
            logger.error(f"Error dropping null values: {e}")

    def fill_columns(self, col_name: str) -> None:
        try:
            if col_name not in self.df.columns:
                raise ValueError(f"Column {col_name} does not exist in DataFrame")
            
            # Convert start_time and end_time columns to timestamp format
            # and extract Year-Month
            self.df = self.df.withColumn("start_time", to_timestamp("start_time"))
            self.df = self.df.withColumn("end_time", to_timestamp("end_time"))
            self.df = self.df.withColumn("YearMonth", date_format(col("start_time"), "yyyy-MM"))

            # Calculate the monthly average of col_name
            # Use a window to fill missing values
            window_spec = Window.partitionBy("YearMonth")
            self.df = self.df.withColumn("monthly_mean", avg(col(col_name)).over(window_spec))

            # Replace null values with the monthly mean
            self.df = self.df.withColumn(col_name, when(col(col_name).isNull(), col("monthly_mean")).otherwise(col(col_name)))
            self.df = self.df.drop("YearMonth", "monthly_mean")
            logger.success(f"Filled missing values for column {col_name} using monthly mean")

        except Exception as e:
            logger.error(f"Error filling column {col_name}: {e}")

    def get_whiskers(self) -> dict:
        """
        Computes whiskers for outlier detection in columns specified in outlier_column_list.

        whiskers are based on Q1 - 1.5*IQR and Q3 + 1.5*IQR, using approxQuantile with relatErr = 0.01 (1%)

        Returns a dict with column names as keys, and a list of [lower_bound, upper_bound] as values
        """

        whiskers = {}
        try:
            for column in self.outlier_column_list:
                if column not in self.df.columns:
                    logger.warning(f"Column {column} not found in DataFrame, skipping")
                    continue
                # getting Q1 and Q3
                q = self.df.approxQuantile(column, [0.25, 0.75], 0.01)
                if len(q) < 2:
                    logger.warning(f"Not enough data to compute whiskers for {column}, skipping")
                    continue
                Q1, Q3 = q
                IQR = Q3 - Q1
                whiskers[column] = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
            logger.success("Computed whiskers for outlier detection")

        except Exception as e:
            logger.error(f"Error computing whiskers: {e}")

        return whiskers

    def remove_outliers(self, whiskers: dict) -> None:
        try:
            condition = None
            for col_name in self.outlier_column_list:
                if col_name not in whiskers:
                    continue
                lb, ub = whiskers[col_name]
                cond_col = (col(col_name) >= lb) & (col(col_name) <= ub)
                condition = cond_col if condition is None else (condition & cond_col)
            if condition is not None:
                self.df = self.df.filter(condition)
                logger.info(f"Removed outliers based on computed whiskers.....Lines: {self.df.count()}, Columns: {len(self.df.columns)}")

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")

    def time_processing(self) -> None:
        """
        Calculate the duration in minutes and extract time-related features from the DataFrame.
        This includes start hour, day, month, and year.
        """

        try:
            self.df = self.df.withColumn("accident_duration(min)", 
                                         (unix_timestamp("end_time") - unix_timestamp("start_time")) / 60.0)
            self.df = self.df.withColumn("start_hour", hour("start_time")) \
                             .withColumn("start_day", dayofmonth("start_time")) \
                             .withColumn("start_month", month("start_time")) \
                             .withColumn("start_year", year("start_time"))
            self.df = self.df.drop("start_time", "end_time")
            logger.success("Processed time-related features")

        except Exception as e:
            logger.error(f"Error in time processing: {e}")


def upload_to_s3(local_file, s3_uri):
    s3 = boto3.client("s3")
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip("/")
    s3.upload_file(local_file, bucket_name, key)

def transform_data(data_source: str, output_uri: str) -> None:
    """
    This function reads a given CSV dataset, drops unnecessary columns, fills missing values, drops null values, processes time columns, 
    removes outliers, and saves the preprocessed dataset to a given output URI.

    Parameters:
    ----------
    data_source : str
        The path to the CSV dataset to preprocess.
    output_uri : str
        The path to save the preprocessed dataset to.

    Returns:
    -------
    None
    """
    spark = SparkSession.builder \
        .appName("Preprocessing Big Data") \
        .getOrCreate()
    
    local_log_path = "/tmp/preprocessing.log"
    logger.add(local_log_path)

    logger.info("Loading data.....")

    frame = spark.read.csv(data_source, header=True, inferSchema=True)
    logger.success(f"Data loaded.....Lines: {frame.count()}, Columns: {len(frame.columns)}")

    # logger.info("Sampling data.....")
    # frame = frame.sample(withReplacement=False, fraction=0.1).limit(1000)
    # logger.info(f"Data sampled.....Lines: {frame.count()}, Columns: {len(frame.columns)}")

    outlier_column_list = ['distance(mi)', 'temperature(f)', 'pressure(in)', 'visibility(mi)', 'wind_speed(mph)', 'precipitation(in)', 'accident_duration(min)']
    preprocessor = Preprocessing(frame, outlier_column_list)

    preprocessor.drop_columns(column_list=['id', 'end_lat', 'end_lng', 'wind_chill(f)'])

    preprocessor.fill_columns(col_name='precipitation(in)')
    preprocessor.fill_columns(col_name='wind_speed(mph)')

    preprocessor.drop_null_values()

    preprocessor.time_processing()

    whiskers = preprocessor.get_whiskers()
    preprocessor.remove_outliers(whiskers)

    preprocessor.df.coalesce(1).write.parquet(output_uri, mode="overwrite") # Coalesce to 1 partition

    s3_log_path = output_uri.rstrip("/") + "/preprocessing.log"
    upload_to_s3(local_log_path, s3_log_path)
    logger.success(f"Dataset saved to {output_uri}..... WORK COMPLETED !!!!!!")

    spark.stop()
    

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing data using PySpark")
    parser.add_argument("--data_source", type=str, help="Path to the data source")
    parser.add_argument("--output_uri", type=str, help="Path to save the processed data")
    args = parser.parse_args()

    transform_data(args.data_source, args.output_uri)   
    