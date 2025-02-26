import pandas as pd
import numpy as np
from traffic_accident_impact.utils import *

from loguru import logger
from pathlib import Path
import typer

from traffic_accident_impact.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR

app = typer.Typer()


columns_to_drop = ["weather_timestamp", "airport_code", "country", "source", "turning_loop", "street", "city", "county", "zipcode"]
columns_to_drop += ["amenity", "bump", "crossing", "give_way", "junction", "no_exit", "railway", "roundabout", "station", "stop", "traffic_calming", "traffic_signal"]
columns_to_drop += ["sunrise_sunset", "civil_twilight", "nautical_twilight", "astronomical_twilight"]

def encoding(df, column, period):
    df[column] = df[column].astype(float)
    radians = 2 * np.pi * df[column] / period
    df[f'sin_{column}'] = np.sin(radians)
    df[f'cos_{column}'] = np.cos(radians)

class FeatureEngineering:
    def __init__(self, frame):
        self.df = frame.drop(columns=columns_to_drop, axis=1)
        self.df = remove_outliers(self.df, 'accident_duration(min)')

    def cyclic_encoding(self):
        encoding(self.df, "start_lng", 360)
        self.df.drop(columns=["start_lng"], inplace=True)
        logger.success(f"Longitude cyclic encoding done !!!")

        encoding(self.df, "start_hour", 24)
        self.df.drop(columns=["start_hour"], inplace=True)
        logger.success(f"Hour cyclic encoding done !!!")

        encoding(self.df, "start_day", 7)
        self.df.drop(columns=["start_day"], inplace=True)
        logger.success(f"Day cyclic encoding done !!!")

        encoding(self.df, "start_month", 12)
        self.df.drop(columns=["start_month"], inplace=True)
        logger.success(f"Month cyclic encoding done !!!")

        # Wind direction to angles mapping
        wind_to_angle = {
            'north': 0, 'n': 0,
            'nne': 22.5,
            'ne': 45,
            'ene': 67.5,
            'east': 90, 'e': 90,
            'ese': 112.5,
            'se': 135,
            'sse': 157.5,
            'south': 180, 's': 180,
            'ssw': 202.5,
            'sw': 225,
            'wsw': 247.5,
            'west': 270, 'w': 270,
            'wnw': 292.5,
            'nw': 315,
            'nnw': 337.5,
            'calm': 0,
            'variable': 180, 'var': 180  # Variable direction, treat as missing
        }
        self.df['wind_angle'] = self.df['wind_direction'].map(wind_to_angle)
        encoding(self.df, "wind_angle", 360)
        self.df.drop(columns=["wind_angle", "wind_direction"], inplace=True)
        logger.success(f"Wind direction cyclic encoding done !!!")


    def grouping_states(self):
        urban_states = [
        'ny', 'nj', 'ma', 'ct', 'ri', 'pa', 'md', 'de', 'dc', 'fl', 'ca', 
        'il', 'mi', 'oh', 'tx', 'ga', 'nc', 'va', 'wa', 'az', 'nv', 'in'
    ]
        rural_states = [
        'me', 'nh', 'vt', 'wv', 'ky', 'tn', 'sc', 'al', 'ms', 'ar', 'la', 
        'ok', 'mo', 'mn', 'wi', 'ia', 'ne', 'nd', 'sd', 'ks', 'wy', 'mt', 
        'id', 'ut', 'nm', 'co', 'or'
    ]
        def assign_category(state):
            if state in urban_states:
                return 'urban'
            elif state in rural_states:
                return 'rural'
            else:
                return 'unknown'
        self.df['state_group'] = self.df["state"].apply(assign_category)
        self.df.drop(columns=["state"], inplace=True)
        logger.success(f"State grouping done !!!")


    def grouping_weathers(self):
        weather_groups = {
            # clear
            'clear': 'clear', 'fair': 'clear',
            
            # cloudy
            'partly cloudy': 'cloudy', 'mostly cloudy': 'cloudy', 'scattered clouds': 'cloudy',
            'overcast': 'cloudy', 'cloudy': 'cloudy',
            
            # precipitation
            'rain': 'precipitation', 'light rain': 'precipitation', 'heavy rain': 'precipitation',
            'rain shower': 'precipitation', 'rain showers': 'precipitation', 
            'light rain shower': 'precipitation', 'light rain showers': 'precipitation',
            'light rain with thunder': 'precipitation', 'showers in the vicinity': 'precipitation',
            'drizzle': 'precipitation', 'light drizzle': 'precipitation', 'heavy drizzle': 'precipitation',
            'light freezing drizzle': 'precipitation', 'thunderstorm': 'precipitation',
            'thunderstorms and rain': 'precipitation', 'light thunderstorms and rain': 'precipitation',
            'heavy thunderstorms and rain': 'precipitation', 't-storm': 'precipitation',
            'heavy t-storm': 'precipitation', 'thunder': 'precipitation', 
            'thunder in the vicinity': 'precipitation', 'snow': 'precipitation', 
            'light snow': 'precipitation', 'light snow shower': 'precipitation',
            'light snow showers': 'precipitation', 'light snow grains': 'precipitation', 
            'blowing snow': 'precipitation', 'squalls': 'precipitation', 'sleet': 'precipitation', 
            'light sleet': 'precipitation', 'rain and sleet': 'precipitation', 
            'light ice pellets': 'precipitation', 'light freezing rain': 'precipitation', 
            'freezing rain': 'precipitation', 'light snow and sleet': 'precipitation', 
            'wintry mix': 'precipitation', 'thunder / wintry mix': 'precipitation',
            
            # obscured
            'fog': 'obscured', 'patches of fog': 'obscured', 'shallow fog': 'obscured', 
            'mist': 'obscured', 'haze': 'obscured', 'light haze': 'obscured', 'smoke': 'obscured',
            'blowing dust': 'obscured', 'widespread dust': 'obscured', 
            'sand / dust whirlwinds': 'obscured', 'sand / dust whirls nearby': 'obscured',
            'tornado': 'obscured', 'funnel cloud': 'obscured', 'volcanic ash': 'obscured', 
            'hail': 'obscured', 'small hail': 'obscured', 'n/a precipitation': 'obscured'
        }
        
        self.df['weather_group'] = self.df["weather_condition"].map(weather_groups).fillna('unknown')
        self.df.drop(columns=["weather_condition"], inplace=True)
        logger.success(f"Weather grouping done !!!")

    

@app.command(help="Feature engineering process to transform, group, and reduce the features")
def transform_save(
    data_path: Path=typer.Option(PROCESSED_DATA_DIR / "final" / "text_processed_data.parquet",
                                help="Path to the parquet file from the previous preprocessing tasks"),
    output_path: Path=typer.Option(PROCESSED_DATA_DIR / "final" / "final_processed_data.parquet",
                                        help="Path to save the final dataset")
):
    logger.add(LOGS_DIR / "feature_engineering.log")

    logger.info("Loading data...")
    df = pd.read_parquet(data_path)
    logger.success(f"Data loaded !!!")

    logger.info("Transforming data...")
    transformer = FeatureEngineering(df)
    transformer.cyclic_encoding()
    transformer.grouping_states()
    transformer.grouping_weathers()
    logger.info(f"Final dataset shape ----> || Lines: {transformer.df.shape[0]} || Features: {transformer.df.shape[1]}")
    logger.info(f"List of features ----> {list(transformer.df.columns)}")

    transformer.df.to_parquet(output_path, engine='pyarrow')
    logger.info(f"Data saved to ----> {output_path}")

    logger.success(f"Feature engineering done !!!")


if __name__ == "__main__":
    app()



    

