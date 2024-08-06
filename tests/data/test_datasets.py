import os
import pandas as pd 
import pytest
from typing import Tuple, List

def get_dataset(raw: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads and returns the train and test datasets based on the 'raw' parameter.

    Parameters:
    raw (bool): If True, reads the raw datasets. Otherwise, reads the processed datasets.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test datasets.
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data', 'raw' if raw else 'processed'))
    return (
        pd.read_csv(os.path.join(base_path, 'train.csv')),
        pd.read_csv(os.path.join(base_path, 'test.csv'))
    )
# Fixture to provide the dataset for the tests
@pytest.fixture
def raw_datasets():
    return get_dataset(raw=True)

@pytest.fixture
def processed_datasets():
    return get_dataset(raw=False)

def test_datasets_shape(raw_datasets, processed_datasets):
    """Test that both the raw and processed datasets have the expected shape."""
    for train, test in [raw_datasets, processed_datasets]:
        assert train.shape[0] > 0, "Train dataset should have at least one row."
        assert train.shape[1] > 0, "Train dataset should have at least one column."
        assert test.shape[0] > 0, "Test dataset should have at least one row."
        assert test.shape[1] > 0, "Test dataset should have at least one column."

def test_no_missing_values_in_target_and_text(raw_datasets, processed_datasets):
    """Test that the 'target' and 'text' columns have no missing values in both datasets."""
    columns_to_check = ['target', 'text']
    for train, test in [processed_datasets]:
        for column in columns_to_check:
            assert not train[column].isnull().values.any(), f"Train dataset should not have missing values in {column} column."
            # assert not test[column].isnull().values.any(), f"Test dataset should not have missing values in {column} column."


def test_expected_columns(raw_datasets, processed_datasets):
    """Test that the dataset contains the expected columns."""
    expected_columns = ['id', "keyword", 'text', 'target', 'location']
    for train, test in [raw_datasets, processed_datasets]:
        assert all(column in train.columns for column in expected_columns), "Train dataset is missing expected columns."
        # assert all(column in test.columns for column in expected_columns), "Test dataset is missing expected columns."

def test_column_types(raw_datasets, processed_datasets):
    """Test that the dataset columns have the expected types."""
    expected_types = {
        'target': 'int64',  
        'id': 'int64',
        'text': 'object',
        'keyword': 'object',
        'location': 'object'

    }
    for train, test in [raw_datasets, processed_datasets]:
        print("Train column types:", train.dtypes)
        print("Test column types:", test.dtypes)
        for column, expected_type in expected_types.items():
            assert train[column].dtype == expected_type, f"Train column {column} should be of type {expected_type}."
            # assert test[column].dtype == expected_type, f"Test column {column} should be of type {expected_type}."


def test_value_ranges(raw_datasets, processed_datasets):
    """Test that numeric columns have values within expected ranges."""
    for train, test in [raw_datasets, processed_datasets]:
        assert train['target'].isin([0, 1]).all(), "Train column 'target' should contain only 0 or 1."
        assert train['target'].isin([0, 1]).all(), "Train column 'target' should contain only 0 or 1."
