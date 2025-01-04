import pandas as pd
import matplotlib.pyplot as plt
import os
from openai import OpenAI

class Preprocessing:
    def __init__(self, frame: pd.DataFrame, outlier_column_list: list[str]) -> None:
        self.df = frame
        self.df.columns =  self.df.columns.str.lower()
        self.outlier_column_list = outlier_column_list
        # whiskers = self.get_whiskers(outlier_column_list)


    def drop_columns(self, column_list: list[str]) -> None:
        self.df = self.df.drop(columns= column_list)

    def drop_null_values(self) -> None:
        self.df = self.df.dropna().reset_index(drop=True)


    def fill_columns(self, col_name: str) -> None:
        self.df['start_time'] = pd.to_datetime(self.df['start_time'], format='mixed')
        self.df['end_time'] = pd.to_datetime(self.df['end_time'], format='mixed')

        # Extract the year and month for each row
        self.df['YearMonth'] = self.df['start_time'].dt.to_period('M')
        
        # Calculate the average for each YearMonth group
        monthly_means = self.df.groupby('YearMonth')[col_name].transform('mean')
        
        # Replace missing values with the corresponding monthly mean
        self.df[col_name] = self.df[col_name].fillna(monthly_means)
        
        # Drop the temporary YearMonth column
        self.df.drop(columns=['YearMonth'], inplace=True)


    def get_whiskers(self) -> dict:
        output = {}
        for column in self.outlier_column_list:
            box = plt.boxplot(self.df[column], showfliers=False)
            # fig, ax = plt.subplots(1, 1)
            # box = ax.boxplot(self.df[column], showfliers = False)
            lower_whisker = box['whiskers'][0].get_ydata()[1]
            upper_whisker = box['whiskers'][1].get_ydata()[1]
            plt.close()
            output[column] = [lower_whisker, upper_whisker]
        return output
    
    def remove_outliers(self, whiskers: dict) -> None:
        for col_name in self.outlier_column_list:
            self.df = self.df[
                (self.df[col_name] >= whiskers[col_name][0]) & (self.df[col_name] <= whiskers[col_name][1])
            ].reset_index(drop=True)

    def plot_boxplot(self, feature_name: str, fliers: bool) -> None:
        fig, ax = plt.subplots(1, 1)
        box = ax.boxplot(self.df[feature_name], showfliers = fliers)
        ax.set_title(feature_name)
        plt.show()
        plt.close()


    def time_processing(self) -> None:
        self.df['accident_duration(min)'] = (self.df['end_time'] - self.df['start_time']).dt.total_seconds() / 60.0
        self.df['start_hour'] = self.df['start_time'].dt.hour
        self.df['start_day'] = self.df['start_time'].dt.day
        self.df['start_month'] = self.df['start_time'].dt.month
        self.df['start_year'] = self.df['start_time'].dt.year
        self.df = self.df.drop(columns=['start_time', 'end_time'])
    

    def embedding(self) -> None:
        client = OpenAI(api_key=os.environ.get('OPEN_API_KEY'))
        response = client.embeddings.create(
        input = self.df['description'],
        model = "text-embedding-ada-002"
    )
        self.df['description'] = response.data
        self.df['description'] = self.df['description'].map(lambda x: x.embedding)



if __name__ == "__main__":
    print(f"Loading data.....\n")
    frame = pd.read_csv('./data/US_Accidents_March23.csv')
    outlier_column_list=['distance(mi)', 'temperature(f)', 'pressure(in)', 'visibility(mi)', 'wind_speed(mph)', 'precipitation(in)', 'accident_duration(min)']
    print(f"Data loaded.....Lines: {frame.shape[0]}, Columns: {frame.shape[1]}\n")

    preprocessor = Preprocessing(frame, outlier_column_list)

    preprocessor.drop_columns(column_list=['id', 'end_lat', 'end_lng', 'wind_chill(f)'])
    print(f"Unnecessary columns dropped.....Lines: {preprocessor.df.shape[0]}, Columns: {preprocessor.df.shape[1]}\n")

    preprocessor.fill_columns(col_name='precipitation(in)')
    preprocessor.fill_columns(col_name='wind_speed(mph)')
    print(f"Missing values filled.....Lines: {preprocessor.df.shape[0]}, Columns: {preprocessor.df.shape[1]}\n")

    preprocessor.drop_null_values()
    print(f"Null values dropped.....Lines: {preprocessor.df.shape[0]}, Columns: {preprocessor.df.shape[1]}\n")

    preprocessor.time_processing()
    print(f"Time processing done.....Lines: {preprocessor.df.shape[0]}, Columns: {preprocessor.df.shape[1]}\n")

    whiskers = preprocessor.get_whiskers()
    print(whiskers)
    preprocessor.remove_outliers(whiskers)
    print(f"Outliers removed.....Lines: {preprocessor.df.shape[0]}, Columns: {preprocessor.df.shape[1]}\n")

    

    preprocessor.df.to_csv('./data/final/data_cleaned.csv', index=False)
    print(f"Data saved.....")

        
        



    