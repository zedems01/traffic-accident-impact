from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, mean as spark_mean, when, avg, hour, dayofmonth, month, year, unix_timestamp
from pyspark.sql.window import Window
import argparse
from colorama import Fore, Style, init

# init(autoreset=True)

# def format_text(text:str, color: str="green", style: str="bold"):
#     """
#     Format text with a specified color and style using colorama.

#     Args:
#     ----
#         text (str): The text to format.
#         color (str): The desired color of the text. Options include: "red", "green",
#                      "yellow", "blue", "magenta", "cyan", "white", "default".
#                      Defaults to "green".
#         style (str): The desired style of the text. Options include: "bold",
#                      "underline", "default". Defaults to "bold".

#     Returns:
#     -------
#         str: The formatted text with the chosen color and style applied.
#     """

#     colors = {
#         "default": "",
#         "red": Fore.RED,
#         "green": Fore.GREEN,
#         "yellow": Fore.YELLOW,
#         "blue": Fore.BLUE,
#         "magenta": Fore.MAGENTA,
#         "cyan": Fore.CYAN,
#         "white": Fore.WHITE,
#     }
    
#     styles = {
#         "default": "",
#         "bold": Style.BRIGHT,
#         "underline": "\033[4m",
#     }
    
#     chosen_color = colors.get(color.lower(), colors["default"])
#     chosen_style = styles.get(style.lower(), styles["default"])
    
#     return f"{chosen_style}{chosen_color}{text}{Style.RESET_ALL}"


class Preprocessing:
    def __init__(self, df: DataFrame, outlier_column_list: list[str]) -> None:
        self.df = df
        for c in self.df.columns:
            self.df = self.df.withColumnRenamed(c, c.lower())
        self.outlier_column_list = outlier_column_list
        
    def drop_columns(self, column_list: list[str]) -> None:
        self.df = self.df.drop(*column_list)

    def drop_null_values(self) -> None:
        self.df = self.df.na.drop()

    def fill_columns(self, col_name: str) -> None:
        # Convert start_time and end_time columns to timestamp format
        self.df = self.df.withColumn("start_time", to_timestamp("start_time"))
        self.df = self.df.withColumn("end_time", to_timestamp("end_time"))

        # Extract Year-Month
        self.df = self.df.withColumn("YearMonth", date_format(col("start_time"), "yyyy-MM"))

        # Calculate the monthly average of col_name
        # Use a window to fill missing values
        window_spec = Window.partitionBy("YearMonth")
        self.df = self.df.withColumn("monthly_mean", avg(col(col_name)).over(window_spec))
        
        # Replace null values with the monthly mean
        self.df = self.df.withColumn(col_name, when(col(col_name).isNull(), col("monthly_mean")).otherwise(col(col_name)))
        
        self.df = self.df.drop("YearMonth", "monthly_mean")

    def get_whiskers(self) -> dict:
        # Determine whiskers from quantiles.
        # Standard whiskers are based on Q1 - 1.5*IQR and Q3 + 1.5*IQR.
        # approxQuantile provides an approximation with param relatErr = 0.01 (1%)
        
        whiskers = {}
        for column in self.outlier_column_list:
            # On récupère Q1 et Q3
            q = self.df.approxQuantile(column, [0.25, 0.75], 0.01)
            if len(q) < 2:
                # Skip if not possible
                continue
            Q1, Q3 = q
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            whiskers[column] = [lower_bound, upper_bound]
        return whiskers

    def remove_outliers(self, whiskers: dict) -> None:
        condition = None
        for col_name in self.outlier_column_list:
            lb, ub = whiskers[col_name]
            cond_col = (col(col_name) >= lb) & (col(col_name) <= ub)
            condition = cond_col if condition is None else (condition & cond_col)
        
        if condition is not None:
            self.df = self.df.filter(condition)


    def time_processing(self) -> None:
        # Calculate the duration in minutes
        # (end_time - start_time) in seconds, then divide by 60
        # spark_diff = unix_timestamp(end_time) - unix_timestamp(start_time)
        self.df = self.df.withColumn("accident_duration(min)", 
                                     (unix_timestamp("end_time") - unix_timestamp("start_time")) / 60.0)
        
        self.df = self.df.withColumn("start_hour", hour("start_time")) \
                         .withColumn("start_day", dayofmonth("start_time")) \
                         .withColumn("start_month", month("start_time")) \
                         .withColumn("start_year", year("start_time"))

        self.df = self.df.drop("start_time", "end_time")


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

    print(format_text("\n\nLoading data.....\n\n", color="green"))
    frame = spark.read.csv(data_source, header=True, inferSchema=True)
    print(format_text(f"\n\nData loaded.....Lines: {frame.count()}, Columns: {len(frame.columns)}\n\n", color="green"))

    # print(format_text("Sampling data.....\n\n", color="green"))
    # frame = frame.sample(withReplacement=False, fraction=0.1).limit(1000)
    # print(format_text(f"Data sampled.....Lines: {frame.count()}, Columns: {len(frame.columns)}\n\n", color="green"))

    outlier_column_list = ['distance(mi)', 'temperature(f)', 'pressure(in)', 'visibility(mi)', 'wind_speed(mph)', 'precipitation(in)', 'accident_duration(min)']
    preprocessor = Preprocessing(frame, outlier_column_list)

    # Drop columns
    preprocessor.drop_columns(column_list=['id', 'end_lat', 'end_lng', 'wind_chill(f)'])
    print(format_text(f"\n\nUnnecessary columns dropped.....Lines: {preprocessor.df.count()}, Columns: {len(preprocessor.df.columns)}\n\n", color="green"))


    # Fill missing values
    preprocessor.fill_columns(col_name='precipitation(in)')
    preprocessor.fill_columns(col_name='wind_speed(mph)')
    print(format_text(f"\n\nMissing values filled.....Lines: {preprocessor.df.count()}, Columns: {len(preprocessor.df.columns)}\n\n", color="green"))

    # Drop null values
    preprocessor.drop_null_values()
    print(format_text(f"\n\nNaN values dropped.....Lines: {preprocessor.df.count()}, Columns: {len(preprocessor.df.columns)}\n", color="green"))

    # Time processing
    preprocessor.time_processing()
    print(format_text(f"\n\nTime processing done.....Lines: {preprocessor.df.count()}, Columns: {len(preprocessor.df.columns)}\n\n", color="green"))


    # Get whiskers for outlier removal
    whiskers = preprocessor.get_whiskers()
    print(whiskers)
    preprocessor.remove_outliers(whiskers)
    print(format_text(f"\n\nOutliers removed.....Lines: {preprocessor.df.count()}, Columns: {len(preprocessor.df.columns)}\n\n", color="green"))

    preprocessor.df.coalesce(1).write.parquet(output_uri, mode="overwrite") # Coalesce to 1 partition

    print(format_text(f"\n\nDataset saved to {output_uri}\n\n", color="green"))
    print(format_text("\n\nWORK COMPLETED !!!!!!\n\n", color="green"))

    spark.stop()
    

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data using PySpark")
    parser.add_argument("--data_source", type=str, help="Path to the data source")
    parser.add_argument("--output_uri", type=str, help="Path to save the processed data")
    args = parser.parse_args()

    transform_data(args.data_source, args.output_uri)   
    