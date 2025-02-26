import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import pickle
from traffic_accident_impact.utils import *
import multiprocessing

from loguru import logger
from pathlib import Path
import typer

from traffic_accident_impact.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR

app = typer.Typer()


class ModelTraining:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame

    def data_preparation(self):
        logger.info("Starting data preparation...")

        self.X = self.df.drop('accident_duration(min)', axis=1)
        self.y = self.df['accident_duration(min)']
        
        numerical_cols = self.X.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.to_list()
        logger.info(f"NUMERICAL COLUMNS ARE: {numerical_cols}")
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.to_list()
        logger.info(f"CATEGORICAL COLUMNS ARE: {self.categorical_cols}")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='error'), self.categorical_cols)
            ])

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(self.X, self.y, test_size=0.2)              # X_test = 20%
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25)   # X_train = 60%, X_val = 20%
    

        logger.success(f"Data preparation done !!!")



    def randomized_searchCV(
            self,
            n_iter: int = 20,
            nb_cv: int = 3,
            n_jobs: int = -1,
            scoring = 'neg_mean_squared_error',
    ):
        
        """Perform randomized search cross-validation to find the best model and hyperparameters.

            This function iterates over a set of models and their respective hyperparameter
            distributions to perform randomized search cross-validation. It identifies the best
            model based on the adjusted R² score and stores the results for each model.

            Parameters
            ----------
            n_iter : int, optional
                Number of parameter settings sampled during the randomized search (default is 20).
            nb_cv : int, optional
                Number of cross-validation folds (default is 3).
            n_jobs : int, optional
                Number of jobs to run in parallel (default is -1, which uses all processors).
            scoring : str, optional
                Scoring metric used for evaluation (default is 'neg_mean_squared_error').

            Returns
            -------
            estimator
                The best model found during randomized search.
        """


        models = {
            # 'Random Forest': RandomForestRegressor(),
            'CatBoost': CatBoostRegressor(silent=True),
            # 'LightGBM': lgb.LGBMRegressor(verbose=-1),
            # 'XGBoost': xgb.XGBRegressor(verbosity=0)
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
            # 'LightGBM': {
            #     'model__n_estimators': [100, 200, 300],
            #     'model__learning_rate': [0.01, 0.05, 0.1],
            #     'model__max_depth': [6, 8, 12]
            # },
            # 'XGBoost': {
            #     'model__n_estimators': [100, 200, 300],
            #     'model__learning_rate': [0.01, 0.05, 0.1],
            #     'model__max_depth': [4, 6, 10]
            # }
        }

        self.best_r2 = float('-inf')
        self.results_train = {}
        logger.info("Starting randomized search...")

        for model_name, model in models.items():
            logger.info(f"Training {model_name} model...")

            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])

            n_jobs = max(1, multiprocessing.cpu_count() - 1)
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions[model_name],
                n_iter=20,
                cv=nb_cv,
                n_jobs=n_jobs,
                scoring=scoring,
                # random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            y_preds = random_search.best_estimator_.predict(self.X_val)

            logger.success(f"Search for {model_name} model done !")

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            mse = -random_search.best_score_
            val_r2_adjusted = adjusted_r2_score(self.y_val, y_preds, self.X_val.shape[1])
            logger.info(f"MSE ----> {mse:.3f}")
            logger.info(f"Adjusted R² ----> {val_r2_adjusted:.3f}")
            self.results_train[model_name] = {
                'best_model': best_model,
                'best_params': best_params,
                'mse': mse,
                'adj_r2': val_r2_adjusted  
            }
            if val_r2_adjusted > self.best_r2:
                self.best_r2 = val_r2_adjusted
                self.best_model = best_model
                self.best_params = best_params
                self.best_model_name = model_name

        logger.success("Randomized search CV done !!!")
        logger.info(f"Best model from randomized search CV ----> {self.best_model_name}\
                    \nBest Adjusted R² from randomized search CV ----> {self.best_r2:.3f}\n")



    def train_best_model(self):
        n_features_test = self.X_test.shape[1]
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        logger.info("Training the best model...")
        self.best_model.fit(X_combined, y_combined)
        logger.success("Best model training done !!!")

        self.test_r2 = adjusted_r2_score(self.y_test, self.best_model.predict(self.X_test), n_features_test)

        logger.info(f"Best model's Adjusted R² on the test set ----> {self.test_r2:.3f}\n")





@app.command(help="Train a model and save the best model")
def train_save(
    parquet_file_path: Path=typer.Option(PROCESSED_DATA_DIR / "final" / "final_processed_data.parquet",
                                           help="Path to the parquet file from the previous preprocessing tasks"),
    frac: float=typer.Option(1.0, 
                               help="Fraction of the data to use for training")
) -> None:
    
    """
    Trains a model with the best parameters and saves

    Parameters
    ----------
    parquet_file_path : str
        Path to the parquet file from the preprocessing tasks
    output_file_name : str
        Name of the output file
    frac : float
        Fraction of the data to use for training

    Returns
    -------
    None
    """
    

    # logger.info("Loading data...")
    df0 = pd.read_parquet(parquet_file_path)
    df = df0.sample(frac=frac)
    del(df0)

    Trainer = ModelTraining(frame=df)
    logger.add(LOGS_DIR / f"training-v1-frac-{frac}.log")
    logger.success("Data loaded !!!")



    Trainer.data_preparation()
    Trainer.randomized_searchCV()
    Trainer.train_best_model()

    name = f'training-v1-{Trainer.best_model_name}-frac-{frac}.pkl'  
    output_file_path = MODELS_DIR / name
    pickle.dump(Trainer.best_model, open(output_file_path, 'wb'))
    logger.success("Model saved !!!")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Training models and saving the best model")
    # parser.add_argument('--parquet_file_path', '-p', type=str, help='Path to the parquet file from the preprocessing tasks')
    # parser.add_argument('--output_file_name', '-o', type=str, help='Name of the output file')
    # parser.add_argument('--cv_method', '-v', type=str, help="Cross-validation method grid_searchCV: grid or randomized_searchCV: random")
    # args = parser.parse_args()
    # train_save(args.parquet_file_path, args.output_file_name, args.cv_method)
    # typer.run(train_save)
    app()
