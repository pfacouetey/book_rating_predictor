import os
import gc
import random
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.transformers.utils.utils import (
    MAX_SCRAPING_ATTEMPTS, BATCH_SIZE, SAVE_INTERVAL,
    load_progress, process_batch, save_progress, text_cleaner, publisher_matcher, subject_matcher, SEED,
    add_gaussian_noise, TEST_SIZE, BOOK_RATING_THRESHOLD
)

SCRAPING_FILENAME = Path(__file__).parent.parent.parent / "data" / "scraping_progress.csv"
SCRAPED3_FILENAME = Path(__file__).parent.parent.parent / "data" / "scraped3_copy.csv"
SCRAPED2_FILENAME = Path(__file__).parent.parent.parent / "data" / "scraped2_copy.csv"
SCRAPED_FILENAME = Path(__file__).parent.parent.parent / "data" / "scraped_copy.csv"

np.random.seed(SEED)
random.seed(SEED)

async def transform_books_dataset(
        books_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """
    Transforms the books dataset by performing the following operations:
    1. Dropping duplicate, missing or incorrect ISBN13 entries.
    2. Dropping books with 0 as average_rating.
    3. Getting some general information on books from a saved progress file if available.
    4. Getting some general information on books through Open Library API if the progress file is not available.
    5. Matching books' publishers and subjects.
    6. Splitting the dataset into standardized training and testing sets.

    :param books_df: A pandas DataFrame containing the books original dataset.
    :return: dict[str, dict[str, pd.DataFrame]] | None: A dictionary containing the structured
        data for original and noisy datasets, each with standardized training and testing
        features and target. Returns None if any input DataFrame is empty.

    Raises:
        ValueError: If `books_df` is empty.
    """

    if books_df.empty:
        logging.error("Input books dataset is empty...")
        return None

    logging.info("Dropping books with duplicate, missing or incorrect isbn13...")
    books_df = books_df.drop_duplicates(subset=["isbn13"])
    books_df = books_df.dropna(subset=["isbn13"])
    books_df = books_df.loc[books_df["isbn13"].str.isdigit()]

    logging.info("Dropping books which have 0 as average_rating...")
    books_df = books_df.mask(books_df["average_rating"] == 0).dropna()

    if os.path.exists(SCRAPED_FILENAME):
        logging.info(f"Getting some general information on books from {SCRAPED_FILENAME}...")
        books_results_df = pd.read_csv(filepath_or_buffer=SCRAPED_FILENAME, dtype=str)

    else:
        logging.info("Getting some general information on books through Open Library API...")

        batches_results = []
        all_isbns = books_df["isbn13"].tolist()
        scraping_attempt = 0

        # Will be run multiple times because some books information might not be found at the first attempt
        while scraping_attempt <= MAX_SCRAPING_ATTEMPTS:

            processed_isbns, processed_books_df = load_progress()
            remaining_isbns = list(set(all_isbns) - processed_isbns)
            num_books = len(remaining_isbns)

            for start_index in range(0, num_books, BATCH_SIZE):
                end_index = min(start_index + BATCH_SIZE, num_books)
                batch_isbns = remaining_isbns[start_index:end_index]
                print(f"Processing batch {start_index // BATCH_SIZE + 1} of {-(-num_books // BATCH_SIZE)}...")

                async with aiohttp.ClientSession() as session:
                    batch_results = await process_batch(
                        session=session, isbns_to_process=batch_isbns, publishers_to_process=[],
                        authors_to_process=[],
                    )
                if batch_results:
                    batches_results.append(pd.concat(batch_results, ignore_index=True))
                    processed_books_df = [processed_books_df, pd.concat(batch_results, ignore_index=True)]
                    processed_isbns.update(batch_isbns)
                # Save progress periodically
                if (start_index // BATCH_SIZE + 1) % SAVE_INTERVAL == 0:
                    save_progress(results=batch_results)
                # Wait to ensure we're within rate limits
                await asyncio.sleep(300) # Wait 5 minutes between batches

            scraping_attempt += 1
            gc.collect()

        books_results_df = pd.concat(batches_results, ignore_index=True)
        for scraped_csv in ["scraped.csv", "scraped_copy.csv"]:
            books_results_df.to_csv(
                path_or_buf=Path(__file__).parent.parent.parent / "data" / scraped_csv,
                index=False,
            )
        del batches_results

    logging.info("Correcting books info...")
    numeric_cols = [
        "num_pages", "revisions_count", "ratings_count", "want_to_read_count", "currently_reading_count",
        "already_read_count"
    ]
    books_results_df[numeric_cols] = books_results_df[numeric_cols].astype(float)
    books_results_df.rename(
        columns={
            "publisher": "publisher_scraped",
            "num_pages": "num_pages_scraped",
            "ratings_count": "rating_count_scraped",
        },
        inplace=True,
    )
    cleaned_books_df = books_df.merge(right=books_results_df,how="left", on="isbn13",)

    logging.info("Fixing and/or removing 0 ratings_count...")
    cleaned_books_df.loc[:, "ratings_count"] = cleaned_books_df.apply(
        lambda row: row["rating_count_scraped"] if row["ratings_count"] == 0 else row["ratings_count"],
        axis=1,
    )

    logging.info("Fixing and/or removing 0 num_pages...")
    cleaned_books_df.loc[:, "num_pages"] = cleaned_books_df.apply(
        lambda row: row["num_pages_scraped"] if row["num_pages"] == 0 else row["num_pages"],
        axis=1,
    )

    logging.info("Fixing publisher name, then getting current total books count for each publisher...")
    cleaned_books_df.loc[:, "publisher"] = cleaned_books_df.apply(
        lambda row: row["publisher_scraped"] if pd.notna(row["publisher_scraped"]) else row["publisher"],
        axis=1,
    )
    if os.path.exists(SCRAPED2_FILENAME):
        logging.info(f"Getting some info on books' publishers from {SCRAPED2_FILENAME}...")
        publishers_results_df = pd.read_csv(filepath_or_buffer=SCRAPED2_FILENAME, dtype=str)
        publishers_results_df["books_count"] = publishers_results_df["books_count"].astype(float)

    else:
        logging.info("Getting some info on books' publishers through Open Library API...")

        batches_results = []
        remaining_publishers = cleaned_books_df["publisher"].unique().tolist()
        num_publishers = len(remaining_publishers)

        for start_index in range(0, num_publishers, BATCH_SIZE):
            end_index = min(start_index + BATCH_SIZE, num_publishers)
            batch_publishers = remaining_publishers[start_index:end_index]
            print(f"Processing batch {start_index // BATCH_SIZE + 1} of {-(-num_publishers // BATCH_SIZE)}...")

            async with aiohttp.ClientSession() as session:
                batch_results = await process_batch(
                    session=session, isbns_to_process=[], publishers_to_process=batch_publishers,
                    authors_to_process=[],
                )
            if batch_results:
                batches_results.append(pd.concat(batch_results, ignore_index=True))
            # Wait to ensure we're within rate limits
            await asyncio.sleep(300) # Wait 5 minutes between batches

        publishers_results_df = pd.concat(batches_results, ignore_index=True)
        for scraped2_csv in ["scraped2.csv", "scraped2_copy.csv"]:
            publishers_results_df.to_csv(
                path_or_buf=Path(__file__).parent.parent.parent / "data" / scraped2_csv,
                index=False,
            )
        del batches_results

    publishers_results_df["cleaned_publisher"] = publishers_results_df["publisher"].apply(text_cleaner)
    publishers_results_df["books_count"] = publishers_results_df.groupby("cleaned_publisher")["books_count"].transform("mean")
    publishers_results_df["choices_for_publisher"] = publishers_results_df["cleaned_publisher"].apply(
        lambda x: [y for y in publishers_results_df["cleaned_publisher"].tolist() if y!=x]
    )
    publishers_results_df["similar_publishers"] = publishers_results_df.apply(
        lambda row: publisher_matcher(
            cleaned_publisher=row["cleaned_publisher"], choices_for_publisher=row["choices_for_publisher"]
        ),
        axis=1,
    )
    publishers_results_df["publisher_score"] = publishers_results_df.apply(
        lambda row: np.mean(
            publishers_results_df.loc[
                publishers_results_df["cleaned_publisher"].isin(row["similar_publishers"]), "books_count"],
        ),
        axis=1,
    )
    cleaned_books_df = cleaned_books_df.merge(
        right=publishers_results_df[["publisher", "publisher_score"]], how="left", on="publisher",
    )

    logging.info("Fixing missing authors, then getting authors' score...")
    cleaned_books_df.loc[:, "authors"] = cleaned_books_df.apply(
        lambda row: row["main_authors"] if pd.notna(row["main_authors"]) else row["authors"].split("/")[0],
        axis=1,
    )
    if os.path.exists(SCRAPED3_FILENAME):
        logging.info(f"Getting some info on books' authors from {SCRAPED3_FILENAME}...")
        authors_results_df = pd.read_csv(filepath_or_buffer=SCRAPED3_FILENAME, dtype=str)
        authors_results_df["author_score"] = authors_results_df["author_score"].astype(float)

    else:
        logging.info("Getting some info on books' authors through Open Library API...")

        batches_results = []
        remaining_authors = cleaned_books_df["authors"].unique().tolist()
        num_authors = len(remaining_authors)

        for start_index in range(0, num_authors, BATCH_SIZE):
            end_index = min(start_index + BATCH_SIZE, num_authors)
            batch_authors = remaining_authors[start_index:end_index]
            print(f"Processing batch {start_index // BATCH_SIZE + 1} of {-(-num_authors // BATCH_SIZE)}...")

            async with aiohttp.ClientSession() as session:
                batch_results = await process_batch(
                    session=session, isbns_to_process=[], publishers_to_process=[],
                    authors_to_process=batch_authors,
                )
            if batch_results:
                batches_results.append(pd.concat(batch_results, ignore_index=True))
            # Wait to ensure we're within rate limits
            await asyncio.sleep(300) # Wait 5 minutes between batches

        authors_results_df = pd.concat(batches_results, ignore_index=True)
        for scraped3_csv in ["scraped3.csv", "scraped3_copy.csv"]:
            authors_results_df.to_csv(
                path_or_buf=Path(__file__).parent.parent.parent / "data" / scraped3_csv,
                index=False,
            )
        del batches_results

    cleaned_books_df = cleaned_books_df.merge(
        right=authors_results_df, how="left", left_on="authors", right_on="author",
    )

    logging.info("Cleaning, then computing similarity between books' subjects...")
    cleaned_books_df["cleaned_subjects"] = cleaned_books_df["subjects"].fillna("unknown").astype(str).apply(text_cleaner)
    subjects_similarities_df = subject_matcher(subjects=cleaned_books_df["cleaned_subjects"].tolist())
    cleaned_books_df = pd.concat([cleaned_books_df, subjects_similarities_df], axis=1)

    logging.info("Dropping unnecessary columns...")
    cleaned_books_df.drop(
        columns=[
            "publisher_scraped", "num_pages_scraped", "rating_count_scraped", "main_authors",
            "rating_count_scraped", "num_pages_scraped", "publisher", "publisher_scraped",
            "main_authors", "authors", "author", "cleaned_subjects", "isbn13", "subjects",
        ],
        inplace=True,
    )
    del subjects_similarities_df, books_results_df, books_df, publishers_results_df, authors_results_df
    gc.collect()

    logging.info("Replacing any missing values by 0, and dropping duplicate rows...")
    cleaned_books_df.fillna(value=0, inplace=True)
    cleaned_books_df.drop_duplicates(inplace=True)

    logging.info("Converting average_rating into a categorical variable...")
    cleaned_books_df["average_rating_category"] = cleaned_books_df["average_rating"].apply(
        lambda x: "low" if x <= BOOK_RATING_THRESHOLD else "high"
    )
    cleaned_books_df.drop(columns=["average_rating"], inplace=True)

    logging.info("Splitting data into training, and test sets...")
    target_df = cleaned_books_df[["average_rating_category"]]
    features_df = cleaned_books_df[cleaned_books_df.columns.drop("average_rating_category")]
    train_features_df, test_features_df, train_target_df, test_target_df = train_test_split(
        features_df, target_df,
        test_size=TEST_SIZE,
        random_state=123,
        )

    logging.info("Removing features with no variation or zero standard deviation...")
    mask_train = (train_features_df.nunique() > 1) & (train_features_df.std() >= 0.8)
    train_features_df = train_features_df.loc[:, mask_train]
    train_features_names = train_features_df.columns.tolist()
    test_features_df = test_features_df[train_features_names]

    logging.info("Adding Gaussian noise to training set...")
    noisy_train_features_df = add_gaussian_noise(features_df=train_features_df)

    logging.info("Standardizing features...")
    scaler1, scaler2 = StandardScaler(), StandardScaler()
    data = {
        "train": {
            "features": {
                "original": pd.DataFrame(
                    data=scaler1.fit_transform(train_features_df),
                    columns=train_features_df.columns,
                ),
                "noise": pd.DataFrame(
                    data=scaler2.fit_transform(noisy_train_features_df),
                    columns=noisy_train_features_df.columns,
                ),
            },
            "targets": train_target_df,
        },
        "test": {
            "features": {
                "original": pd.DataFrame(
                    data=scaler1.transform(test_features_df),
                    columns=test_features_df.columns,
                ),
                "noise": pd.DataFrame(
                    data=scaler2.transform(test_features_df),
                    columns=test_features_df.columns,
                ),
            },
            "targets": test_target_df,
        }
    }

    return data
