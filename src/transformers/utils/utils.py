import io
import os
import re
import nltk
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from fuzzywuzzy import process
from nltk import word_tokenize
from nltk.corpus import stopwords
from typing import Optional, List
from aiohttp import ClientTimeout
nltk.download("punkt", quiet=True)
from circuitbreaker import circuit
from aiolimiter import AsyncLimiter
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

SCRAPING_FILENAME = Path(__file__).parent.parent.parent.parent / "data" / "scraping_progress.csv"
MAX_REQUESTS = AsyncLimiter(max_rate=100, time_period=4.5 * 60) # A limit of 100 requests each 5 min imposed by Open Library API
HUGGING_FACE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
STOP_WORDS = set(stopwords.words("english"))
OPEN_LIBRARY_URL = "https://openlibrary.org"
MAX_SCRAPING_ATTEMPTS = 2
BOOK_RATING_THRESHOLD = 4 # It's being used to categorize average_rating values into 2 classes
SAVE_INTERVAL = 1 # Save progress after processing each batch
BATCH_SIZE = 200 # 200, 100 # Process no more than 100 books/publishers, and 200 authors by batch
MAX_RETRIES = 10
TEST_SIZE = 0.3
TIMEOUT = 600 # 600, 300 # Use a timeout of 300 or 600 seconds for each request
SEED = 123


def is_missing(data: dict[str, any]) -> bool:
    """
    Checks if all values in the given dictionary are missing (NaN).

    Args:
        data (dict[str, any]): The dictionary to check.

    Returns:
        bool: True if all values are missing, False otherwise.
    """

    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")

    return all(pd.isna(value) for value in data.values())

@circuit(failure_threshold=10, recovery_timeout=300)
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_result(is_missing))
async def fetch_data(
        session: aiohttp.ClientSession,
        url: str,
) -> Optional[dict]:
    """
    Fetches data asynchronously from a given URL using an aiohttp ClientSession.

    This function uses exponential backoff retry strategy to handle transient
    errors during data fetching. It sends a GET request to the specified URL.

    Args:
        session (aiohttp.ClientSession): The aiohttp ClientSession to use for the request.
        url (str): The URL to fetch data from.

    Returns:
        Optional[dict]: A dictionary containing the JSON response data if the request is
        successful and status code is 200, otherwise returns None.

    Raises:
        asyncio.TimeoutError: If the request times out.
        Exception: If any other unexpected error occurs during the request.
    """

    async with MAX_REQUESTS:
        try:
            async with session.get(url, timeout=ClientTimeout(total=TIMEOUT)) as response:

                if response.status == 200:
                    return await response.json()

                logging.error(f"Error retrieving data from {url}: {await response.text()}")

        except asyncio.TimeoutError:
            logging.error(f"Timeout error for {url}")

        except Exception as e:
            logging.error(f"Unexpected error for {url}: {str(e)}")

        return None

async def book_scraper(
        session: aiohttp.ClientSession,
        isbn: str,
) -> Optional[pd.DataFrame]:
    """
    Scrapes book information from Open Library API.

    Args:
        session (aiohttp.ClientSession): The aiohttp ClientSession to use for the request.
        isbn (str): The ISBN-13 of the book to fetch information for.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing book information if the request is
        successful and status code is 200, otherwise returns None.

    Raises:
        asyncio.TimeoutError: If the request times out.
        Exception: If any other unexpected error occurs during the request.
    """

    book_info = {"isbn13": isbn,}
    urls = [
        f"{OPEN_LIBRARY_URL}/api/books?bibkeys=ISBN:{isbn}&jscmd=data&format=json",
        None,
        None,
        None
    ]

    for i, url in enumerate(urls):
        if url is None:
            continue

        data = await fetch_data(session, url)
        if not data:
            if i == 0:  # If the first request fails, we can't proceed
                return None
            continue  # For other requests, we can continue with partial data

        if i == 0:
            new_data = data.get(f"ISBN:{isbn}", {})
            book_info.update({
                "main_authors": new_data.get("authors", [{}])[0].get("name", None),
                "publisher": new_data.get("publishers", [{}])[0].get("name", None),
                "num_pages": np.nan if new_data.get("number_of_pages", None) is None else int(new_data.get("number_of_pages")),
            })
            id_v1 = new_data.get("identifiers", {}).get("openlibrary", [""])[0]
            if id_v1:
                urls[1] = f"{OPEN_LIBRARY_URL}/books/{id_v1}.json"
            else:
                break # If we can't get id_v1, we can't proceed with other requests

        elif i == 1:
            book_info.update({
                "subjects": ", ".join(data.get("subjects", None)),
                "revisions_count": np.nan if data.get("latest_revision", None) is None else int(data.get("latest_revision"))
            })
            id_v2 = data.get("works", [{}])[0].get("key", "").split("/")[-1]
            if id_v2:
                urls[2] = f"{OPEN_LIBRARY_URL}/works/{id_v2}/ratings.json"
                urls[3] = f"{OPEN_LIBRARY_URL}/works/{id_v2}/bookshelves.json"
            else:
                break # If we can't get id_v2, we can't proceed with other requests

        elif i == 2:
            summary = data.get("summary", {})
            book_info["ratings_count"] = np.nan if summary.get("count", None) is None else int(summary.get("count"))

        elif i == 3:
            counts = data.get("counts", {})
            book_info.update({
                "want_to_read_count": np.nan if counts.get("want_to_read", None) is None else int(counts.get("want_to_read")),
                "currently_reading_count": np.nan if counts.get("currently_reading", None) is None else int(counts.get("currently_reading")),
                "already_read_count": np.nan if counts.get("already_read", None) is None else int(counts.get("already_read")),
            })

    return pd.DataFrame([book_info])

async def author_scorer(
        session: aiohttp.ClientSession,
        author: str,
) -> Optional[float]:

    if author.strip() == "":
        return None

    new_author = author.replace(" ", "%20")  # Replace spaces with '%20' for URL encoding
    url = f"https://openlibrary.org/search/authors.json?q={new_author}"
    try:
        async with session.get(url, timeout=ClientTimeout(total=TIMEOUT)) as response:

            if response.status == 200:
                data = await response.json()
                docs = data.get("docs", [{}])

                if docs:
                    author_scores = [
                        float(doc.get("work_count")) for doc in docs if doc.get("work_count", None) is not None
                    ]
                    author_final_score = np.sum(author_scores)
                    return pd.DataFrame([{"author": author, "author_score": author_final_score}])

                return pd.DataFrame([{"author": author, "author_score": np.nan}])

            logging.error(f"Error retrieving data from {url}: {await response.text()}")

    except asyncio.TimeoutError:
        logging.error(f"Timeout error for {url}")

    except Exception as e:
        logging.error(f"Unexpected error for {url}: {str(e)}")

    return None

async def publisher_scraper(
        session: aiohttp.ClientSession,
        publisher: str,
) -> Optional[pd.DataFrame]:
    """
    Scrapes the total number of books published by a given publisher from Open Library API.

    Args:
        session (aiohttp.ClientSession): The aiohttp ClientSession to use for the request.
        publisher (str): The name of the publisher to fetch the total number of books for.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the publisher name and the total number of books
        published by it if the request is successful and status code is 200, otherwise returns None.

    Raises:
        asyncio.TimeoutError: If the request times out.
        Exception: If any other unexpected error occurs during the request.
    """

    if publisher.strip() == "":
        return None

    url = f"https://openlibrary.org/search/publishers.json?q={publisher}"
    try:
        async with session.get(url, timeout=ClientTimeout(total=TIMEOUT)) as response:

            if response.status == 200:
                raw_content = await response.text()
                if raw_content:
                    parsed_content = BeautifulSoup(raw_content, "html.parser")
                    raw_books_count = [span.text for span in parsed_content.find_all("span", class_="count")]
                    books_count = [
                        int(re.findall(r'\d+', item)[0]) for item in raw_books_count if "book" in item.lower()
                    ]
                    return pd.DataFrame([{"publisher": publisher, "books_count": np.sum(books_count)}])

            logging.error(f"Error retrieving data from {url}: {await response.text()}")

    except asyncio.TimeoutError:
        logging.error(f"Timeout error for {url}")

    except Exception as e:
        logging.error(f"Unexpected error for {url}: {str(e)}")

    return None

async def process_batch(
        session: aiohttp.ClientSession,
        isbns_to_process: List[str],
        publishers_to_process: List[str],
        authors_to_process: List[str],
) -> List[pd.DataFrame] | None:
    """
    Process a batch of ISBNs, publishers, or authors concurrently.

    Args:
        session (aiohttp.ClientSession): The aiohttp ClientSession to use for the requests.
        isbns_to_process (List[str]): The list of ISBNs to fetch book information for.
        publishers_to_process (List[str]): The list of publishers to fetch the total number of books published by.
        authors_to_process (List[str]): The list of authors to fetch their author scores.

    Returns:
        List[pd.DataFrame] | None: A list of DataFrames containing the book information, publisher counts, or author scores
        for the given batch, if all requests are successful and status code is 200, otherwise returns None.

    Raises:
        asyncio.TimeoutError: If any of the requests times out.
        Exception: If any other unexpected error occurs during the requests.
    """

    tasks = []

    if len(isbns_to_process) > 0 and len(publishers_to_process) == 0 and len(authors_to_process) == 0:
        tasks = [book_scraper(session=session, isbn=isbn_13) for isbn_13 in isbns_to_process]

    elif len(publishers_to_process) > 0 and len(isbns_to_process) == 0 and len(authors_to_process) == 0:
        tasks = [publisher_scraper(session=session, publisher=publisher) for publisher in publishers_to_process]

    elif len(authors_to_process) > 0 and len(isbns_to_process) == 0 and len(publishers_to_process) == 0:
        tasks = [author_scorer(session=session, author=author) for author in authors_to_process]

    if len(tasks) > 0:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [df for df in results if isinstance(df, pd.DataFrame)]
        return valid_results

    return None

def save_progress(
        results: List[pd.DataFrame],
):
    """
    Saves the progress of the scraping process to a CSV file.

    Args:
        results (List[pd.DataFrame]): The list of DataFrames containing scraped books information.
    """

    if isinstance(results, List) and len(results) > 0:

        results_df = pd.concat(results, ignore_index=True)

        if os.path.exists(SCRAPING_FILENAME):
            existing_df = pd.read_csv(filepath_or_buffer=SCRAPING_FILENAME)
            combined_df = pd.concat([existing_df, results_df]).drop_duplicates(subset=["isbn13"])
            combined_df.to_csv(SCRAPING_FILENAME, index=False)
        else:
            results_df.to_csv(SCRAPING_FILENAME, index=False)
            print(f"Add {len(results)} new records to the CSV scraping_progress...")

        print(f"Progress saved. Processed {len(results)} ISBNs so far.")

    else:
        logging.error("No results to save...")

def load_progress():
    """
    Loads the progress of the scraping process from a CSV file.

    Returns:
        Tuple[set[str], pd.DataFrame]: A tuple containing the set of processed ISBNs and the loaded DataFrame.
    """

    if os.path.exists(SCRAPING_FILENAME):

        with open(SCRAPING_FILENAME, "rb") as file:
            raw_data = file.read()
            decoded_data = raw_data.decode("utf-8", errors="replace")
            processed_books_df = pd.read_csv(io.StringIO(decoded_data))
            processed_books_df["isbn13"] = processed_books_df["isbn13"].astype(str)
            processed_isbns = set(processed_books_df["isbn13"].tolist())
        print(f"Loaded progress. Resuming from {len(processed_isbns)} processed ISBNs...")

        return processed_isbns, processed_books_df

    return set(), pd.DataFrame()

def text_cleaner(raw_text: str) -> str:
    """
    Cleans a given text by removing punctuation, converting to lowercase, and removing stop words.

    Args:
        raw_text (str): The raw text to clean.

    Returns:
        str: The cleaned text.
    """

    tokens = word_tokenize(re.sub(r"[^\w\s]", "", raw_text).lower())
    cleaned_text = [word for word in tokens if word.lower() not in STOP_WORDS]
    return " ".join(cleaned_text)

def publisher_matcher(
        cleaned_publisher: str,
        choices_for_publisher: List[str],
        threshold=95,
) -> List[str]:
    """
    Performs fuzzy matching between a cleaned publisher name and a list of choices for publisher.

    Args:
        cleaned_publisher (str): The cleaned publisher name to match.
        choices_for_publisher (List[str]): The list of choices for publisher.
        threshold (int): The minimum similarity score required for a match.

    Returns:
        List[str]: The best matched choices for the cleaned publisher name.
    """

    best_choices_for_publisher = [cleaned_publisher]

    for scorer in [fuzz.token_sort_ratio, fuzz.partial_ratio, fuzz.ratio]:
        result = process.extractOne(
            query=cleaned_publisher,
            choices=choices_for_publisher,
            scorer=scorer,
            score_cutoff=threshold
        )
        if result and result[1] >= threshold:
            best_choices_for_publisher.append(result[0])

    return best_choices_for_publisher

def subject_matcher(
        subjects: List[str],
) -> pd.DataFrame:
    """
    Performs semantic matching between a list of subjects and returns a DataFrame with similarity scores.

    Args:
        subjects (List[str]): The list of subjects to match.

    Returns:
        pd.DataFrame: A DataFrame containing the similarity scores between the subjects.
    """

    sentences = [subject.replace("unknown", "") for subject in subjects]
    non_empty_sentences = [sentence for sentence in sentences if sentence]
    non_empty_embeddings = HUGGING_FACE_MODEL.encode(sentences=non_empty_sentences, show_progress_bar=True)
    non_empty_similarities = HUGGING_FACE_MODEL.similarity(non_empty_embeddings, non_empty_embeddings).numpy()

    index_mapping = {
        old_index: new_index for new_index, old_index in enumerate(
            [_index for _index, sentence in enumerate(sentences) if sentence])
    }
    full_similarities = np.zeros((len(sentences), len(sentences)))
    for index1, sentence1 in enumerate(sentences):
        for index2, sentence2 in enumerate(sentences):
            if sentence1 and sentence2:
                full_similarities[index1, index2] = non_empty_similarities[index_mapping[index1], index_mapping[index2]]
            if not sentence1 and not sentence2 and index1 == index2:
                full_similarities[index1, index2] = 1.0

    return pd.DataFrame(data=full_similarities, columns=[f"subject_{i+1}" for i in range(len(sentences))])


def add_gaussian_noise(
        features_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Adds Gaussian noise to the input DataFrame.

    This function takes a DataFrame of features and applies Gaussian noise to each
    value. It generates noise using a normal distribution with a mean of 0.0 and a
    standard deviation of 1.0. If the input DataFrame is empty, it logs a warning
    and returns None. The resulting DataFrame, with added Gaussian noise, is
    returned.

    Args:
        features_df (pd.DataFrame): The input DataFrame containing feature values
        whose entries will have Gaussian noise applied.

    Returns:
        pd.DataFrame | None: A DataFrame with Gaussian noise added to its values,
        or None if the input DataFrame is empty.

    Raises:
        This function does not explicitly raise any errors, but unexpected issues
        could arise due to DataFrame processing or noise generation.
    """

    if features_df.empty:
        logging.warning("No features to add Gaussian noise to.")
        return None

    logging.info("Generating Gaussian noise...")
    noise_df = pd.DataFrame(
        data=np.random.normal(
            loc=0.0,
            scale=0.5,
            size=(
                features_df.shape[0],
                features_df.shape[1],
            )
        ),
        columns=features_df.columns.tolist(),
        index=features_df.index,
    )
    logging.info("Gaussian noise generation completed successfully.")
    return features_df + noise_df
