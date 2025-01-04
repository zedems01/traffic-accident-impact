import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import pickle
from utils import *
import argparse
import multiprocessing
import os

class ModelTraining:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame

    def data_preparation(self):
        print(format_text(f"\nStarting data preparation......\n"))

        self.df.drop(columns=["visibility(mi)"], inplace=True)
        # int64, int32 to category
        self.df[['severity', 'start_hour', 'start_day', 'start_month', 'start_year']] = \
            self.df[['severity', 'start_hour', 'start_day', 'start_month', 'start_year']].astype('category')
        
        # object to category
        obj_cols = self.df.select_dtypes(include=['object', 'bool']).columns.to_list()
        self.df[obj_cols] = self.df[obj_cols].astype('category')

        # normalize categorical cols
        df_cat_cols = self.df.select_dtypes(include=['category']).columns 
        # self.df[df_cat_cols] = self.df[df_cat_cols].map(lambda col: col.lower().strip())
        self.df[df_cat_cols] = self.df[df_cat_cols].apply(lambda col: col.astype(str).str.lower().str.strip())
        # print(format_text(f"\nNormalization done !!!\n", color='red', style='default'))


        self.X = self.df.drop('accident_duration(min)', axis=1)
        self.y = self.df['accident_duration(min)']
        
        numerical_cols = self.X.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.to_list()
        categorical_cols = self.X.select_dtypes(include=['category']).columns.to_list()

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) 

        self.preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='error'), categorical_cols)
        ])

        print(format_text(f"\nData preparation done !!!\n"))


    def ramdom_forest_feature_selection(self):
        return 0
    

    def grid_searchCV(self):
        print(format_text(f"\nStarting grid search......\n"))

        models = {
            # 'Random Forest': RandomForestRegressor(),
            'CatBoost': CatBoostRegressor(silent=True),
            'LightGBM': lgb.LGBMRegressor(),
            'XGBoost': xgb.XGBRegressor()
        }
        param_grid = {
        # 'Random Forest': {
        #     'model__n_estimators': [100, 200],
        #     'model__max_depth': [10, 20],
        #     'model__min_samples_split': [2, 5],
        # },
        'CatBoost': {
            'model__iterations': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__depth': [6, 10]
        },
        'LightGBM': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [6, 10]
        },
        'XGBoost': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [6, 10]
        }
}
        
        self.best_score = float('-inf')
        for model_name, model in models.items():
            print(format_text(f"\nTraining {model_name} model......", color='red', style='default'))
            
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid[model_name],
                cv=3, n_jobs=n_jobs,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(self.X_train, self.y_train)
            
            preds = grid_search.best_estimator_.predict(self.X_val)

            # val_mse = pow(root_mean_squared_error(self.y_val, preds) ,2)
            # print(format_text(f"Best {model_name} model MSE: {val_mse:.3f}\n", color='red', style='default'))

            n_features = self.X_train.shape[1]
            val_r2_adjusted = adjusted_r2_score(self.y_val, preds, n_features)
            print(format_text(f"Best {model_name} model Adjusted R²: {val_r2_adjusted:.3f}\n", color='red', style='default'))

            if val_r2_adjusted > self.best_score:
                self.best_score = val_r2_adjusted
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name

        print(format_text(f"\nGrid search done !!!"))
        print(format_text(f"Best model: {self.best_model_name}\nAdjusted R²: {self.best_score:.3f}\n"))

        return self.best_model


    def randomized_searchCV(self):
        print(format_text(f"\nStarting RandomizedSearchCV......\n"))

        models = {
            # 'Random Forest': RandomForestRegressor(),
            'CatBoost': CatBoostRegressor(silent=True),
            'LightGBM': lgb.LGBMRegressor(),
            'XGBoost': xgb.XGBRegressor()
        }
        param_distributions = {
            # 'Random Forest': {
            #     'model__n_estimators': [100, 200, 300, 500],
            #     'model__max_depth': [10, 20, 30, None],
            #     'model__min_samples_split': [2, 5, 10],
            # },
            'CatBoost': {
                'model__iterations': [100, 200, 500],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__depth': [4, 6, 10]
            },
            'LightGBM': {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [6, 8, 12]
            },
            'XGBoost': {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [4, 6, 10]
            }
        }

        self.best_score = float('-inf')
        for model_name, model in models.items():
            print(format_text(f"\nTraining {model_name} model......", color='red', style='default'))

            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])

            n_jobs = max(1, multiprocessing.cpu_count() - 1)
            randomized_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions[model_name],
                n_iter=20,
                cv=3,
                n_jobs=n_jobs,
                scoring='neg_mean_squared_error',
                random_state=42
            )
            randomized_search.fit(self.X_train, self.y_train)

            preds = randomized_search.best_estimator_.predict(self.X_val)

            # val_mse = pow(root_mean_squared_error(self.y_val, preds), 2)
            # print(format_text(f"Best {model_name} model MSE: {val_mse:.3f}\n", color='red', style='default'))

            n_features = self.X_train.shape[1]
            val_r2_adjusted = adjusted_r2_score(self.y_val, preds, n_features)
            print(format_text(f"Best {model_name} model Adjusted R²: {val_r2_adjusted:.3f}\n", color='red', style='default'))

            if val_r2_adjusted > self.best_score:
                self.best_score = val_r2_adjusted
                self.best_model = randomized_search.best_estimator_
                self.best_model_name = model_name

        print(format_text(f"\nRandomized search done !!!"))
        print(format_text(f"Best model: {self.best_model_name}\nAdjusted R²: {self.best_score:.3f}\n"))

        return self.best_model


    def train_best_model(self):
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        print(format_text(f"\nTraining the best model......"))
        self.best_model.fit(X_combined, y_combined)
        print(format_text(f"Best model training done !!!\n", color='red', style='default'))
        # self.mse = pow(root_mean_squared_error(self.y_test, self.best_model.predict(self.X_test)) ,2)
        self.r2_adjusted = adjusted_r2_score(self.y_test, self.best_model.predict(self.X_test), self.X_test.shape[1])
        print(format_text(f"Best model Adjusted R² on the test set: {self.r2_adjusted:.3f}\n"))

def train_save(parquet_file_path: str, output_file_name: str, cv_method: str='grid') -> None:
        """
        This function trains a model with the best parameters and saves it to a file

        Parameters
        ----------
        parquet_file_path : str
            Path to the parquet file from the preprocessing tasks
        output_file_name : str
            Name of the output file
        cv_method : str
            Cross-validation method grid_searchCV 'grid' or randomized_searchCV 'random'

        Returns
        -------
        None
        """
        print(format_text(f"\nLoading data......\n"))
        df = pd.read_parquet(parquet_file_path)
        df = df.sample(frac=0.8)
        print(format_text(f"\nData loaded !!!\n"))

        Trainer = ModelTraining(frame=df)

        Trainer.data_preparation()
        if cv_method == 'grid':
            Trainer.grid_searchCV()
        else:
            Trainer.grid_searchCV()

        Trainer.train_best_model()
        output_file_path = "./data/model"
        pickle.dump(Trainer.best_model, open(os.path.join(output_file_path, output_file_name), 'wb'))
        print(format_text(f"\nModel saved !!!\n"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training models and saving the best model")
    parser.add_argument('--parquet_file_path', '-p', type=str, help='Path to the parquet file from the preprocessing tasks')
    parser.add_argument('--output_file_name', '-o', type=str, help='Name of the output file')
    parser.add_argument('--cv_method', '-v', type=str, help="Cross-validation method grid_searchCV: grid or randomized_searchCV: random")
    args = parser.parse_args()
    train_save(args.parquet_file_path, args.output_file_name, args.cv_method)
