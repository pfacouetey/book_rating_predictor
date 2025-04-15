import pytest
import pandera as pa

from src.loaders.data_loader import load_original_books_dataset
from tests.loaders.fixtures.fixtures import OriginalBookSchema


def test_load_original_books_dataset():
    """
    Tests the functionality of the `load_original_books_dataset` function by validating
    the DataFrame against the provided schema.

    This test ensures that the dataset loaded by the function adheres to the specified
    schema, which includes the expected column names and data types.

    :return: None, as this is a test function that performs assertions directly.
    """
    books_df = load_original_books_dataset()
    try:
        OriginalBookSchema.validate(books_df)
    except pa.errors.SchemaError as e:
        pytest.fail(f"DataFrame validation failed: {e}")
