import logging
import pandas as pd
from pathlib import Path


def load_original_books_dataset() -> pd.DataFrame|None:
    """
    Loads and processes the original books dataset from a predefined CSV file location. The function reads the
    dataset, performs data type conversions for specific columns, and filters out relevant attributes for
    further processing. If the dataset is empty or cannot be located, a log message is recorded, and the function
    returns `None`.

    :return: A pandas DataFrame containing the processed books dataset with specific attributes
        or `None` if the dataset fails to load.
    :rtype: pd.DataFrame | None
    """
    books_csv_path = Path(__file__).parent.parent.parent / "data" / "books.csv"
    books_df = pd.read_csv(filepath_or_buffer=books_csv_path, dtype=str)

    if books_df.empty:
        logging.error(f"Failed to load books dataset. File not found: {books_csv_path}")
        return None

    integer_cols = ["num_pages", "ratings_count", "text_reviews_count"]
    books_df[integer_cols] = books_df[integer_cols].astype(int)

    books_df["publication_year"] = books_df["publication_date"].str.split("/").str[-1].astype(int)
    books_df["average_rating"] = books_df["average_rating"].str.replace(",", ".").astype(float)
    cols_to_keep = [
        "isbn13",
        "authors",
        "average_rating",
        "publication_year",
        "publisher",
        "num_pages",
        "ratings_count",
        "text_reviews_count",
    ]

    return books_df[cols_to_keep]
