import pytest
import asyncio
import pandas as pd


@pytest.fixture()
def isbn():
    return "9780976694007"

@pytest.fixture()
def expected_book_info_df(isbn):
    return pd.DataFrame(
        {
            "isbn13": isbn,
            "main_authors": "Dave Thomas",
            "num_pages": 558,
            "publisher": "Pragmatic Bookshelf",
            "subjects": "Web site development -- Handbooks, manuals, etc, Ruby (Computer program language) -- Handbooks, manuals, etc",
            "revisions_count": 16,
            "ratings_count": 0,
            "want_to_read_count": 4,
            "currently_reading_count": 0,
            "already_read_count": 1,
        },
        index=[0],
    )

@pytest.fixture()
def publisher():
    return "Nimble Books"

@pytest.fixture()
def expected_publisher_books_count_df(publisher):
    return pd.DataFrame(
        {
            "publisher": publisher,
            "books_count": 181,
        },
        index=[0],
    )

@pytest.fixture()
def author():
    return "J.K. Rowling"

@pytest.fixture()
def expected_author_score_df(author):
    return pd.DataFrame(
        {
            "author": author,
            "author_score": 411.0,
        },
        index=[0],
    )

@pytest.fixture()
def expected_book_info(expected_book_info_df):
    return [expected_book_info_df]

@pytest.fixture()
def expected_publisher_books_count(expected_publisher_books_count_df):
    return [expected_publisher_books_count_df]

@pytest.fixture()
def expected_author_score(expected_author_score_df):
    return [expected_author_score_df]

@pytest.fixture(scope="function")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()



