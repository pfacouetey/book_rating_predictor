import pytest
import aiohttp
import pandas as pd

from src.transformers.utils.utils import book_scraper, publisher_scraper, author_scorer, process_batch
from tests.transformers.fixtures.fixtures import (
    event_loop,
    isbn, publisher, author,
    expected_book_info_df, expected_publisher_books_count_df, expected_author_score_df,
    expected_book_info, expected_publisher_books_count, expected_author_score,
)


@pytest.mark.asyncio
async def test_book_scraper(
        isbn,
        expected_book_info_df,
        event_loop,
):
    """
    This test verifies that the book_scraper function returns the expected book information dataframe.

    :param isbn: A string representing the 13-digit ISBN of the book to fetch information for.
    :param expected_book_info_df: A Pandas DataFrame containing the expected information of the book for validation purposes.
    :param event_loop: The event loop instance used to manage asynchronous operations in the test.
    :return: None
    """

    async with aiohttp.ClientSession() as session:
        actual_book_info_df = await book_scraper(isbn=isbn, session=session)

    pd.testing.assert_frame_equal(actual_book_info_df, expected_book_info_df, check_like=True, check_exact=True)

@pytest.mark.asyncio
async def test_publisher_scraper(
        publisher,
        expected_publisher_books_count_df,
        event_loop,
):
    """
    This test verifies that the publisher_scraper function returns the expected publisher books count dataframe.

    :param publisher: A string representing the name of the publisher to fetch information for.
    :param expected_publisher_books_count_df: A Pandas DataFrame containing the expected information of the publisher's books for validation purposes.
    :param event_loop: The event loop instance used to manage asynchronous operations in the test.
    :return: None
    """

    async with aiohttp.ClientSession() as session:
        actual_publisher_books_count_df = await publisher_scraper(publisher=publisher, session=session)

    pd.testing.assert_frame_equal(
        actual_publisher_books_count_df, expected_publisher_books_count_df, check_like=True, check_exact=True
    )

@pytest.mark.asyncio
async def test_author_scorer(
        author,
        event_loop,
        expected_author_score_df,
):
    """
    This test verifies that the author_scorer function returns the expected author score dataframe.

    :param author: A string representing the name of the author to fetch information for.
    :param event_loop: The event loop instance used to manage asynchronous operations in the test.
    :param expected_author_score_df: A Pandas DataFrame containing the expected information of the author's score for validation purposes.
    :return: None
    """

    async with aiohttp.ClientSession() as session:
        actual_author_score_df = await author_scorer(session=session, author=author)

    pd.testing.assert_frame_equal(
        actual_author_score_df, expected_author_score_df, check_like=True, check_exact=True
    )

@pytest.mark.asyncio
async def test_process_batch(
        event_loop,
        isbn,
        author,
        publisher,
        expected_book_info,
        expected_author_score,
        expected_publisher_books_count,
):
    """
    This test verifies that the process_batch function processes multiple books, publishers, and authors concurrently.

    :param event_loop:
    :param isbn: A string representing the 13-digit ISBN of the book to fetch information for.
    :param author: A string representing the name of the author to fetch information for.
    :param publisher: A string representing the name of the publisher to fetch information for.
    :param expected_book_info: A list of Pandas DataFrames containing the expected information of the books for validation purposes.
    :param expected_author_score: A list of Pandas DataFrames containing the expected information of the authors' scores for validation purposes.
    :param expected_publisher_books_count: A list of Pandas DataFrames containing the expected information of the publishers' books count for validation purposes.
    :return: None
    """

    async with aiohttp.ClientSession() as session:
        actual_book_info = await process_batch(
            session=session, isbns_to_process=[isbn], publishers_to_process=[], authors_to_process=[],
        )
        actual_publisher_books_count = (
            await process_batch(
                session=session, isbns_to_process=[], publishers_to_process=[publisher],
                authors_to_process=[],
            )
        )
        actual_author_score = await process_batch(
            session=session, isbns_to_process=[], publishers_to_process=[], authors_to_process=[author],
        )

    assert len(actual_book_info) == len(expected_book_info), "Expected {}, but got {}".format(
        expected_book_info, actual_book_info)
    assert len(actual_publisher_books_count) == len(expected_publisher_books_count), "Expected {}, but got {}".format(
        expected_publisher_books_count, actual_publisher_books_count)
    assert len(actual_author_score) == len(expected_author_score), "Expected {}, but got {}".format(
        expected_author_score, actual_author_score)

    pd.testing.assert_frame_equal(actual_book_info[0], expected_book_info[0], check_like=True, check_exact=True)
    pd.testing.assert_frame_equal(
        actual_publisher_books_count[0], expected_publisher_books_count[0], check_like=True, check_exact=True
    )
    pd.testing.assert_frame_equal(
        actual_author_score[0], expected_author_score[0], check_like=True, check_exact=True
    )