import re
import string
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import argparse

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from utils import *


class TextPreprocessing:
    def __init__(self, data: pd.DataFrame, language='english'):
        self.df = data
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def tokenizer(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Stopwords filtration, Lemmatization
        cleaned_tokens = [
            self.lemmatizer.lemmatize(tok) 
            for tok in tokens 
            if tok not in self.stop_words and tok.isalpha()  # Also filter non-alphabetical tokens
        ]
        return cleaned_tokens
    
    def tokenize(self) -> None:
        self.df['tokens'] = self.df['description'].apply(self.tokenizer)


    def trainDoc2Vec(self, vector_size: int=100, window: int=5, min_count: int=1, workers: int=4, dm: int=1, epochs: int=20) -> None:
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
        # Vocabulary construction
        model_doc2vec.build_vocab(tagged_data)

        model_doc2vec.train(
            tagged_data,
            total_examples=model_doc2vec.corpus_count,
            epochs=model_doc2vec.epochs
        )
        print(format_text("Doc2Vec model trained!", color="red"))
        
        vector_df = pd.DataFrame(
            [model_doc2vec.dv[i] for i in range(len(self.df))],  # Vectors' extraction
            columns=[f'vector_{i}' for i in range(model_doc2vec.vector_size)]  # Column names
        )

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(vector_df)

        for i in range(3):
            self.df[f'description_pca{i+1}'] = pca_result[:, i]

        self.df.drop(columns=['description', 'tokens'], inplace=True)


def transform_text_data(parquet_file_path: str, output_file_path: str) -> None:
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

    df = pd.read_parquet(parquet_file_path)
    print(format_text(f"\n\nData loaded !!!\n\n"))

    # df = df.sample(100)

    text_preprocessor = TextPreprocessing(data=df)

    print(format_text("\n\nTokenizing text data....."))
    text_preprocessor.tokenize()
    print(format_text("Tokenization done !!!\n\n"))

    print(format_text("\n\nTraining Doc2Vec embedding....."))
    text_preprocessor.trainDoc2Vec()
    print(format_text("Doc2Vec embedding done !!!\n\n"))

    # text_preprocessor.df.to_csv(output_file_path, index=False)
    text_preprocessor.df.to_parquet(output_file_path, engine='pyarrow')
    print(format_text(f"\n\nData saved !!!\n\n"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Description column using Doc2Vec")
    parser.add_argument("--parquet_file_path", "-p", type=str, help="Path to the parquet file from the spark job")
    parser.add_argument("--output_file_path", "-o", type=str, help="Path to save the output file")
    args = parser.parse_args()

    transform_text_data(args.parquet_file_path, args.output_file_path)
