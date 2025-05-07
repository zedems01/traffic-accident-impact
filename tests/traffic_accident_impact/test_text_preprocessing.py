import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
import numpy as np
from typer.testing import CliRunner
from traffic_accident_impact.text_preprocessing import app as text_preprocessing_app

from src.traffic_accident_impact.text_preprocessing import TextPreprocessing

# Mock nltk.download at the module level for all tests in this file to prevent actual downloads during testing.
@pytest.fixture(autouse=True)
def mock_nltk_download():
    with patch('nltk.download') as mock_download:
        yield mock_download

@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """Creates a sample Pandas DataFrame for text preprocessing tests."""
    data = {
        'severity': [1, 2, 1, 3],
        'start_hour': [10, 14, 10, 20],
        'start_day': [1, 2, 1, 5],
        'start_month': [1, 6, 1, 12],
        'start_year': [2023, 2023, 2023, 2024],
        'description': [
            "A test description with punctuation!",
            "Another one, for testing stopwords like 'the' and 'a'.",
            "  EXTRA SPACES  and mixed CASE.  ",
            "Lemmatization: running runs ran."
        ],
        'visibility(mi)': [10.0, 5.0, 10.0, 8.0], # This column should be dropped
        'some_object_col': ['TypeA', 'TypeB', ' typea ', 'TypeC'],
        'some_bool_col': [True, False, True, True]
    }
    return pd.DataFrame(data)

@pytest.fixture
def text_preprocessor_instance(sample_pandas_df: pd.DataFrame) -> TextPreprocessing:
    """Fixture to get an instance of TextPreprocessing."""
    return TextPreprocessing(data=sample_pandas_df.copy())


# --- Tests for TextPreprocessing class ---

def test_nltk_resources_downloaded(mock_nltk_download):
    """Test that nltk.download is called for the required resources when the module is imported."""
    # We need to trigger the import of text_preprocessing again or check calls if it was already imported.
    expected_calls = [
        call('stopwords'),
        call('punkt_tab'),
        call('wordnet')
    ]
    for expected_call in expected_calls:
        assert expected_call in mock_nltk_download.call_args_list

def test_text_preprocessor_init(text_preprocessor_instance: TextPreprocessing, sample_pandas_df: pd.DataFrame):
    assert text_preprocessor_instance.df.equals(sample_pandas_df)
    assert 'english' in text_preprocessor_instance.stop_words # Check a common stopword
    assert hasattr(text_preprocessor_instance, 'lemmatizer')

def test_categorical_features_normalization(text_preprocessor_instance: TextPreprocessing):
    tp = text_preprocessor_instance
    tp.categorical_features_normalization()
    
    # Check column drop
    assert 'visibility(mi)' not in tp.df.columns
    
    # Check type conversions
    categorical_cols_to_check = ['severity', 'start_hour', 'start_day', 'start_month', 'start_year']
    for col in categorical_cols_to_check:
        assert tp.df[col].dtype == 'category', f"Column {col} should be category type"
    
    object_cols_converted = ['some_object_col', 'some_bool_col']
    for col in object_cols_converted:
         assert tp.df[col].dtype == 'category', f"Column {col} should be category type after object conversion"

    # Check normalization (lowercase, strip)
    # Example: 'some_object_col' had ' typea '
    assert 'typea' in tp.df['some_object_col'].cat.categories
    assert ' typea ' not in tp.df['some_object_col'].cat.categories
    # Check if a numeric-like category column was also normalized
    # The code does: col.astype(str).str.lower().str.strip()
    # So, for 'severity' which is [1,2,1,3], it becomes ['1','2','1','3']
    assert '1' in tp.df['severity'].cat.categories
    assert 1 not in tp.df['severity'].cat.categories 

def test_tokenizer(text_preprocessor_instance: TextPreprocessing):
    tp = text_preprocessor_instance
    
    text1 = "This is a test, with punctuation! And stopwords like the, a, is."
    expected1 = ['test', 'punctuation', 'stopwords']
    assert tp.tokenizer(text1) == expected1
    
    text2 = "  EXTRA SPACES  and runs running ran.  "
    # Lemmatization: runs -> run, running -> run, ran -> run (WordNetLemmatizer default is verb if no pos)
    # Let's check the output from nltk.word_tokenize("runs running ran") -> ['runs', 'running', 'ran']
    # lemmatizer.lemmatize('runs') -> 'run' (noun)
    # lemmatizer.lemmatize('running') -> 'running' (noun)
    # lemmatizer.lemmatize('ran') -> 'ran' (noun)
    expected2 = ['extra', 'spaces', 'runs', 'running', 'ran']
    assert tp.tokenizer(text2) == expected2
    
    text3 = "123 numbers &*^% special characters"
    expected3 = ['numbers', 'special', 'characters']
    assert tp.tokenizer(text3) == expected3
    
    text4 = ""
    expected4 = []
    assert tp.tokenizer(text4) == expected4

    # Test type hint correction: should be list[str]
    assert isinstance(tp.tokenizer(text1), list)
    if tp.tokenizer(text1):
        assert isinstance(tp.tokenizer(text1)[0], str)

def test_tokenize_method(text_preprocessor_instance: TextPreprocessing):
    tp = text_preprocessor_instance
    tp.tokenize() # Applies the tokenizer to the 'description' column
    
    assert 'tokens' in tp.df.columns
    assert isinstance(tp.df['tokens'].iloc[0], list)
    # Check one example from the sample_pandas_df
    # "A test description with punctuation!" -> ["test", "description", "punctuation"]
    assert tp.df['tokens'].iloc[0] == ['test', 'description', 'punctuation']

    # Test if 'description' column does not exist
    df_no_desc = pd.DataFrame({'col1': [1,2]})
    tp_no_desc = TextPreprocessing(data=df_no_desc)
    with patch('loguru.logger.warning') as mock_logger_warning:
        tp_no_desc.tokenize()
        mock_logger_warning.assert_called_once_with("Column 'description' not found in DataFrame")
        assert 'tokens' not in tp_no_desc.df.columns

# --- Tests for trainDoc2Vec ---

@patch('traffic_accident_impact.text_preprocessing.PCA')
@patch('traffic_accident_impact.text_preprocessing.Doc2Vec')
def test_trainDoc2Vec(MockDoc2Vec, MockPCA, text_preprocessor_instance: TextPreprocessing):
    tp = text_preprocessor_instance
    # Ensure 'tokens' column exists
    tp.tokenize()
    assert 'tokens' in tp.df.columns

    # Configure mocks
    mock_doc2vec_model = MagicMock()
    MockDoc2Vec.return_value = mock_doc2vec_model
    # Simulate the model having document vectors; create dummy vectors
    num_rows = len(tp.df)
    vector_size = 100 # from default in trainDoc2Vec
    dummy_vectors = [np.random.rand(vector_size) for _ in range(num_rows)]
    mock_doc2vec_model.dv = MagicMock()
    # Make mock_doc2vec_model.dv act like a list/dict for access by index
    mock_doc2vec_model.dv.__getitem__.side_effect = lambda i: dummy_vectors[i]
    mock_doc2vec_model.vector_size = vector_size # Ensure the model instance has vector_size

    mock_pca_instance = MagicMock()
    MockPCA.return_value = mock_pca_instance
    # PCA result: n_samples x n_components
    dummy_pca_result = np.random.rand(num_rows, 3)
    mock_pca_instance.fit_transform.return_value = dummy_pca_result

    tp.trainDoc2Vec(vector_size=vector_size, epochs=1)

    # Assertions for Doc2Vec
    MockDoc2Vec.assert_called_once_with(
        vector_size=vector_size, window=5, min_count=1, workers=4, dm=1, epochs=1
    )
    mock_doc2vec_model.build_vocab.assert_called_once()
    # Check that tagged_data was passed to build_vocab
    # tagged_data = [TaggedDocument(words=row['tokens'], tags=[idx]) ...]
    assert len(mock_doc2vec_model.build_vocab.call_args[0][0]) == num_rows # First arg, first element of tuple
    assert hasattr(mock_doc2vec_model.build_vocab.call_args[0][0][0], 'words')
    assert hasattr(mock_doc2vec_model.build_vocab.call_args[0][0][0], 'tags')

    mock_doc2vec_model.train.assert_called_once()
    # Check total_examples and epochs in train call
    assert mock_doc2vec_model.train.call_args[1]['total_examples'] == mock_doc2vec_model.corpus_count
    assert mock_doc2vec_model.train.call_args[1]['epochs'] == mock_doc2vec_model.epochs
    
    # Assertions for PCA
    MockPCA.assert_called_once_with(n_components=3)
    # Check that fit_transform was called on a DataFrame of the correct shape (num_rows x vector_size)
    pca_input_arg = mock_pca_instance.fit_transform.call_args[0][0]
    assert isinstance(pca_input_arg, pd.DataFrame)
    assert pca_input_arg.shape == (num_rows, vector_size)
    
    # Check DataFrame modifications
    assert 'description_pca1' in tp.df.columns
    assert 'description_pca2' in tp.df.columns
    assert 'description_pca3' in tp.df.columns
    assert tp.df['description_pca1'].iloc[0] == dummy_pca_result[0, 0]

    assert 'description' not in tp.df.columns
    assert 'tokens' not in tp.df.columns

def test_trainDoc2Vec_no_tokens_column(text_preprocessor_instance: TextPreprocessing):
    tp = text_preprocessor_instance
    # Ensure 'tokens' column does NOT exist
    if 'tokens' in tp.df.columns:
        tp.df.drop(columns=['tokens'], inplace=True)
    
    with pytest.raises(ValueError, match="DataFrame does not contain 'tokens' column"):
        tp.trainDoc2Vec()

# --- Test for transformTextData (CLI command) ---

@patch('pandas.read_parquet')
@patch('traffic_accident_impact.text_preprocessing.TextPreprocessing') # Mock the class itself
@patch('pandas.DataFrame.to_parquet') # Mock the final save operation
@patch('loguru.logger.add') # Mock logger configuration
def test_transformTextData_cli(mock_logger_add, mock_df_to_parquet, MockTextPreprocessing, mock_pd_read_parquet, tmp_path):
    runner = CliRunner()
    
    # Prepare mock behaviors
    mock_input_df = pd.DataFrame({'description': ['text1', 'text2']}) # Dummy input df
    mock_pd_read_parquet.return_value = mock_input_df
    
    mock_tp_instance = MagicMock() # Mock instance of TextPreprocessing
    MockTextPreprocessing.return_value = mock_tp_instance
    final_df_to_save = pd.DataFrame({'processed': [1,2]})
    mock_tp_instance.df = final_df_to_save 

    input_file = tmp_path / "input.parquet"
    output_file = tmp_path / "output.parquet"
    
    # Create a dummy input file for read_parquet
    pd.DataFrame().to_parquet(input_file)

    result = runner.invoke(text_preprocessing_app, [
        "--parquet-file-path", str(input_file),
        "--output-file-path", str(output_file)
    ])

    assert result.exit_code == 0, f"CLI command failed: {result.stdout}"
    
    # Verify mocks
    mock_pd_read_parquet.assert_called_once_with(input_file)
    MockTextPreprocessing.assert_called_once_with(data=mock_input_df)
    
    # Check that methods of TextPreprocessing instance were called
    mock_tp_instance.categorical_features_normalization.assert_called_once()
    mock_tp_instance.tokenize.assert_called_once()
    mock_tp_instance.trainDoc2Vec.assert_called_once()
    
    # Check that the final DataFrame (from the mocked instance) was saved
    mock_df_to_parquet.assert_called_once()
    # The DataFrame instance is the first argument to the globally mocked pandas.DataFrame.to_parquet
    saved_df_instance = mock_df_to_parquet.call_args[0][0]
    output_path_arg = mock_df_to_parquet.call_args[0][1]
    
    assert output_path_arg == output_file
    pd.testing.assert_frame_equal(saved_df_instance, final_df_to_save)

    # Check logger add was called
    mock_logger_add.assert_called()
