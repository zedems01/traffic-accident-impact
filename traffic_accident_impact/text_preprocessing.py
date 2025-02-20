# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#
#
# -----------------------------------------


import re
import string
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import argparse
import typer
from loguru import logger
from pathlib import Path

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from traffic_accident_impact.config import MODELS_DIR, PROCESSED_DATA_DIR, LOGS_DIR




class TextPreprocessing:
    def __init__(self, data: pd.DataFrame, language='english'):
        self.df = data
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def categorical_features_normalization(self) -> None:  
        """
            Drops column "visibility(mi)".
            Converts int64, int32 categorical features to category type.
            Converts object columns to category type.
            Normalizes categorical columns by converting them to lower case and stripping.
        """

        self.df.drop(columns=["visibility(mi)"], inplace=True)
        logger.success(f"Dropped column 'visibility(mi)'...")

        # int64, int32 to category
        self.df[['severity', 'start_hour', 'start_day', 'start_month', 'start_year']] = \
            self.df[['severity', 'start_hour', 'start_day', 'start_month', 'start_year']].astype('category')
        
        # object to category
        obj_cols = self.df.select_dtypes(include=['object', 'bool']).columns.to_list()
        self.df[obj_cols] = self.df[obj_cols].astype('category')
        logger.success(f"Converted object columns to category type...")

        # normalize categorical cols
        df_cat_cols = self.df.select_dtypes(include=['category']).columns 
        self.df[df_cat_cols] = self.df[df_cat_cols].apply(lambda col: col.astype(str).str.lower().str.strip())

        logger.success(f"Categorical columns normalization done!!")

    def tokenizer(self, text: str) -> str:
        try:
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.word_tokenize(text)
            # Stopwords filtration, Lemmatization
            cleaned_tokens = [
                self.lemmatizer.lemmatize(tok) 
                for tok in tokens 
                if tok not in self.stop_words and tok.isalpha()  # Also filter non-alphabetical tokens
            ]
            return cleaned_tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []        
    
    def tokenize(self) -> None:
        logger.info("Applying tokenization...")
        try:
            if 'description' in self.df.columns:
                self.df['tokens'] = self.df['description'].apply(self.tokenizer)
                logger.success("Tokenization applied successfully!!")
            else:
                logger.warning("Column 'description' not found in DataFrame")
        except Exception as e:
            logger.error(f"Error applying tokenization: {e}")


    def trainDoc2Vec(self, vector_size: int=100, window: int=5, min_count: int=1, workers: int=4, dm: int=1, epochs: int=20) -> None:
        logger.info("Training Doc2Vec model...")

        try:
            if 'tokens' not in self.df.columns:
                raise ValueError("DataFrame does not contain 'tokens' column")
            
            tagged_data = [
                TaggedDocument(words=row['tokens'], tags=[idx]) 
                for idx, row in self.df.iterrows()
            ]
            
            model_doc2vec = Doc2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count, 
                workers=workers,
                dm=dm,
                epochs=epochs
            )
            
            model_doc2vec.build_vocab(tagged_data)
            model_doc2vec.train(
                tagged_data,
                total_examples=model_doc2vec.corpus_count,
                epochs=model_doc2vec.epochs
            )
            logger.success("Doc2Vec model trained successfully!!")
            
            vector_df = pd.DataFrame(
                [model_doc2vec.dv[i] for i in range(len(self.df))],
                columns=[f'vector_{i}' for i in range(model_doc2vec.vector_size)]
            )
            
            logger.info("Applying PCA...")
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(vector_df)
            
            for i in range(3):
                self.df[f'description_pca{i+1}'] = pca_result[:, i]
            
            self.df.drop(columns=['description', 'tokens'], inplace=True, errors='ignore')
            logger.success("Applied PCA successfully!!")

            logger.info("Text preprocessing complete!!")

        except Exception as e:
            logger.error(f"Error in trainDoc2Vec: {e}")

app = typer.Typer()

@app.command(help="Preprocess the Description column using Doc2Vec")
def transformTextData(
    parquet_file_path: Path=typer.Argument(PROCESSED_DATA_DIR / "emr-spark-job" / "part-00000-1e8dbbd6-6f6f-4c21-9a0d-4f0752bb4ca0-c000.snappy.parquet",
                                           help="Path to the parquet file from the spark job"),
    output_file_path: Path=typer.Argument(PROCESSED_DATA_DIR / "final" / "final_data.parquet",
                                          help="Path to save the output file")
) -> None:
    
    """
    Transforms text data by tokenizing and generating Doc2Vec embeddings.

    This function reads a parquet file containing text data, tokenizes the 
    text, trains a Doc2Vec model to create embeddings, and saves the 
    transformed data, including PCA components of the embeddings, to a CSV file.

    Args:
    ----
        parquet_file_path (str): The path to the input parquet file containing text data.
        output_file_path (str): The path to save the output CSV file with transformed data.

    Returns:
    --------
        None
    """

    logger.add(LOGS_DIR / "text_preprocessing.log")

    df = pd.read_parquet(parquet_file_path)
    logger.info(f"Data loaded!")

    text_preprocessor = TextPreprocessing(data=df)

    text_preprocessor.categorical_features_normalization()
    text_preprocessor.tokenize()
    text_preprocessor.trainDoc2Vec()

    # text_preprocessor.df.to_csv(output_file_path, index=False)
    text_preprocessor.df.to_parquet(output_file_path, engine='pyarrow')
    logger.info(f"Data saved to {output_file_path}")

if __name__ == "__main__":
    app()
    
