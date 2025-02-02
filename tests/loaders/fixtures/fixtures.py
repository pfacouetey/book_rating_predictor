import pandera as pa
from pandera.typing import Series

class OriginalBookSchema(pa.DataFrameModel):
    """
    A schema for the books' dataset.
    """

    isbn13: Series[str]
    authors: Series[str]
    average_rating: Series[float]
    publication_year: Series[int]
    publisher: Series[str]
    num_pages: Series[int]
    ratings_count: Series[int]
    text_reviews_count: Series[int]

    class Config:
        strict = False
        coerce = True
