import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import pickle
from traffic_accident_impact.utils import *
import multiprocessing
import gc

from loguru import logger
import warnings
from pathlib import Path
import typer

from traffic_accident_impact.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR

app = typer.Typer()




class ModelTraining:
    def __init__(self, frame: pd.DataFrame, frac: float = 1.0):
        self.frac = frac
        self.df = frame.sample(frac=frac)

    def data_preparation(self):
        logger.info("Starting data preparation...")

        self.X = self.df.drop('accident_duration(min)', axis=1)
        self.y = self.df['accident_duration(min)']
        
        numerical_cols = self.X.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.to_list()
        # logger.info(f"NUMERICAL COLUMNS ARE: {numerical_cols}")
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.to_list()
        # logger.info(f"CATEGORICAL COLUMNS ARE: {self.categorical_cols}")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='error'), self.categorical_cols)
            ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)              # X_test = 20%
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25)   # X_train = 60%, X_val = 20%
    

        logger.success(f"Data preparation done !!!")


    def save_training_results(self, best_pipeline, model_name: str, output_path_frac: Path):
        logger.info("Saving training results...")

        # saving best pipeline
        name = f'{model_name}-frac-{self.frac}.pkl'  
        pickle.dump(best_pipeline, open(output_path_frac / name, 'wb'))

        y_pred_train = best_pipeline.best_estimator_.predict(self.X_train)
        y_pred_test = best_pipeline.best_estimator_.predict(self.X_test)

        # scores
        train_r2 = adjusted_r2_score(self.y_train, y_pred_train, self.X_train.shape[1])
        test_r2 = adjusted_r2_score(self.y_test, y_pred_test, self.X_test.shape[1])
        train_rmse = root_mean_squared_error(self.y_train, y_pred_train)
        test_rmse = root_mean_squared_error(self.y_test, y_pred_test)

        logger.info(f"{model_name} model,  frac={self.frac},  Adjusted R² on the train set ----> {train_r2:.3f}")
        logger.info(f"{model_name} model,  frac={self.frac},  Adjusted R² on the test set ----> {test_r2:.3f}")
        logger.info(f"{model_name} model,  frac={self.frac},  RMSE on the train set ----> {train_rmse:.3f}")
        logger.info(f"{model_name} model,  frac={self.frac},  RMSE on the test set ----> {test_rmse:.3f}")


        # feature importance
        get_feature_importance(best_pipeline.best_estimator_, model_name, output_path_frac, self.X_train, self.y_train)

        # residual analysis
        resid_train = get_residuals(self.y_train, y_pred_train, output_path_frac / 'train_residuals.png')
        resid_test = get_residuals(self.y_test, y_pred_test, output_path_frac / 'test_residuals.png')
        logger.info(f"Train set:  {resid_train}")
        logger.info(f"Test set:  {resid_test}")

        logger.success(f"RESULT SAVED !")


    def randomized_searchCV(
            self,
            model_name: str,
            n_iter: int = 20,
            nb_cv: int = 3,
            n_jobs: int = -1,
            scoring = 'neg_root_mean_squared_error'
            
    ):
        
        """Perform randomized search cross-validation to find the best model and hyperparameters.

            This function iterates over a set of models and their respective hyperparameter
            distributions to perform randomized search cross-validation. It identifies the best
            model based on the adjusted R² score and stores the results for each model.

            Parameters
            ----------
            model_name : str
                The name of the model (choices: 'linear', 'randomforest', 'xgboost', 'catboost', 'lightgbm').
            n_iter : int, optional
                Number of parameter settings sampled during the randomized search (default is 20).
            nb_cv : int, optional
                Number of cross-validation folds (default is 3).
            n_jobs : int, optional
                Number of jobs to run in parallel (default is -1, which uses all processors).
            scoring : str, optional
                Scoring metric used for evaluation (default is 'neg_mean_squared_error').
        """


        models = {
            'Linear': LinearRegression(),
            'RandomForest': RandomForestRegressor(),
            'CatBoost': CatBoostRegressor(silent=True),
            'LightGBM': lgb.LGBMRegressor(verbose=-1),
            'XGBoost': xgb.XGBRegressor(verbosity=0)
        }

        param_distributions = {
            'Linear': {},
            'RandomForest': {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
            },
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

        model_type = model_name.lower()
        if "linear" in model_type:
            selected_model_name = 'Linear'
        elif "random" in model_type or "forest" in model_type:
            selected_model_name = 'RandomForest'
        elif "cat" in model_type:
            selected_model_name = 'CatBoost'
        elif "xg" in model_type:
            selected_model_name = 'XGBoost'
        elif "light" in model_type:
            selected_model_name = 'LightGBM'
        else:
            raise ValueError("Model type not recognized. Choose between linear, randomforest, xgboost, catboost, lightgbm")

        model = models[selected_model_name]
        params = param_distributions[selected_model_name]
        output_path_model = MODELS_DIR / f"{selected_model_name}"
        output_path_frac = output_path_model / f"frac-{self.frac}"
        output_path_frac.mkdir(parents=True, exist_ok=True)


        logger.info("Starting randomized search...")

        logger.info(f"Training {selected_model_name} model...")

        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])

        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=n_iter,
            cv=nb_cv,
            n_jobs=n_jobs,
            scoring=scoring,
        )
        random_search.fit(self.X_train, self.y_train)
        logger.success(f"RANDOMIZED SEARCH CV FOR {selected_model_name} MODEL DONE !")

        self.save_training_results(random_search, selected_model_name, output_path_frac)

        



@app.command(help="Train a model and save models")
def train_save(
    parquet_file_path: Path=typer.Option(PROCESSED_DATA_DIR / "final" / "final_processed_data.parquet",
                                           help="Path to the parquet file from the previous preprocessing tasks"),
    frac: float=typer.Option(1.0, 
                               help="Fraction of the data to use for training"),
    model_name: str=typer.Option("linear",
                                 help="Name of the model to train. Choices: linear, randomforest, xgboost, catboost, lightgbm")
) -> None:
    
    """
    Trains a model with the best parameters and saves

    Parameters
    ----------
    parquet_file_path : str
        Path to the parquet file from the preprocessing tasks
    frac : float
        Fraction of the data to use for training
    model_name : str
        Name of the model to train. Choices: 'linear', 'randomforest', 'xgboost', 'catboost', 'lightgbm'
    """
    

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{filename}:{lineno} - {category.__name__}: {message}")
    warnings.showwarning = warning_handler
    
    df = pd.read_parquet(parquet_file_path)

    Trainer = ModelTraining(frame=df, frac=frac)
    del df
    gc.collect()
    logger.add(LOGS_DIR / f"{model_name}-frac-{frac}.log")
    logger.success("DATA LOADED AND SAMPLED !")



    Trainer.data_preparation()
    Trainer.randomized_searchCV(model_name=model_name, nb_cv=5)
    # Trainer.train_best_model()

    # name = f'training-v1-{Trainer.best_model_name}-frac-{frac}.pkl'  
    # output_file_path = MODELS_DIR / name
    # pickle.dump(Trainer.best_model, open(output_file_path, 'wb'))
    # logger.success("Model saved !")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Training models and saving the best model")
    # parser.add_argument('--parquet_file_path', '-p', type=str, help='Path to the parquet file from the preprocessing tasks')
    # parser.add_argument('--output_file_name', '-o', type=str, help='Name of the output file')
    # parser.add_argument('--cv_method', '-v', type=str, help="Cross-validation method grid_searchCV: grid or randomized_searchCV: random")
    # args = parser.parse_args()
    # train_save(args.parquet_file_path, args.output_file_name, args.cv_method)
    # typer.run(train_save)
    app()
