<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Traffic Accident Impact Prediction

## Overview

This project aims to predict the duration of traffic accidents across 49 states in the USA and assess their impact on traffic flow. Using a dataset of approximately 7.7 million accident records (February 2016–March 2023) sourced from real-time traffic incident APIs, the project leverages data preprocessing, feature engineering, and machine learning to enable faster decision-making for road safety and traffic regulation.

The workflow includes distributed preprocessing with PySpark on AWS EMR, text vectorization with Doc2Vec, feature engineering, and model training with five regression algorithms: Linear Regression, Random Forest, XGBoost, CatBoost, and LightGBM. The final output is a set of optimized models capable of forecasting accident duration, along with performance metrics and visualizations.

## Dataset

The dataset is a compilation of car accident records from diverse APIs, capturing data from transportation departments, law enforcement, traffic cameras, and road sensors. It includes features such as accident location, time, weather conditions, and textual descriptions. For more details, visit [the dataset source](https://smoosavi.org/datasets/us_accidents).

## Project Structure

The project is organized into the following phases, each with corresponding scripts:

1. **Preprocessing with PySpark on AWS EMR** (`processing_spark.py`)
   - Handles large-scale data cleaning, null value imputation, outlier removal, and time feature extraction.
   - Output: Preprocessed Parquet file.

2. **Text Processing with Doc2Vec** (`text_preprocessing.py`)
   - Tokenizes and vectorizes accident descriptions using Doc2Vec, followed by PCA for dimensionality reduction.
   - Output: Enriched dataset with text embeddings.

3. **Feature Engineering** (`feature_engineering.py`)
   - Applies cyclic encoding to temporal and spatial features, groups states and weather conditions, and drops redundant columns.
   - Output: Final processed dataset.

4. **Modeling** (`model_training.py` + `utils.py`)
   - Trains five regression models using pipelines with hyperparameter tuning via `RandomizedSearchCV`.
   - Saves models, metrics (adjusted R², RMSE), feature importance, and residual plots for further analysis.
   - Output: Trained model files (`.pkl`), metrics, and visualizations.

5. **Results Analysis**
The final report of the project can be found [here](./src/reports/report.pdf)



## Project Organization

traffic_accident_impact/
├── src/                                    
│   ├── models/                             
│   │   ├── LightGBM/                       
│   │   ├── Linear/                         
│   │   └── XGBoost/                        
│   ├── notebooks/                          
│   ├── reports/                              
│   │   ├── figures/                        
│   │   └── logs/                           
│   ├── modeling/                            
│   │   ├── __init__.py                     
│   │   ├── config.py                       
│   │   ├── feature_engineering.py          
│   │   ├── processing_pandas.py             
│   │   ├── processing_spark.py             
│   │   ├── text_preprocessing.py           
│   │   └── utils.py                        
│   └── .gitkeep
├── tests/                                   
├── .gitignore  
├── .pdm-python  
├── LICENSE  
├── README.md  
├── pdm.lock                                
└── pyproject.toml                          



