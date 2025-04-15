import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score


PENALTY_KIND = None
SOLVER_KIND = "saga"
MAX_ITER = 500
L1_RATIO = 0.5
N_FOLDS = 10
SEED = 123


np.random.seed(SEED)
random.seed(SEED)


def pca_classifier(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "test"] = "train",
        modeling_results: dict[str, object] = None,
) -> dict[str, object] | None:
    if features_df.empty or targets_df.empty:
        logging.error("At least one of input dataFrames is empty ...")
        return None

    if operation_mode == "train":
        logging.info("Creating cross-validation sets...")
        cv = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)

        logging.info("Training a logistic regression model...")
        results = {}
        for target_name in targets_df.columns:
            model = LogisticRegression(penalty=PENALTY_KIND, solver=SOLVER_KIND, max_iter=MAX_ITER, l1_ratio=L1_RATIO)
            scoring = {
                "recall": "recall",
                "roc_auc": "roc_auc",
                "precision": "precision"
            }
            scores = cross_validate(
                estimator=model,
                X=features_df,
                y=targets_df[target_name],
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            model.fit(features_df, targets_df[target_name])
            results[target_name] = {
                "model": model,
                "val_roc_auc": np.mean(scores["test_roc_auc"]),
                "val_recall": np.mean(scores["test_recall"]),
                "val_precision": np.mean(scores["test_precision"]),
                "train_roc_auc": np.mean(scores["train_roc_auc"]),
                "train_recall": np.mean(scores["train_recall"]),
                "train_precision": np.mean(scores["train_precision"]),
            }
        return results

    if operation_mode == "test" and modeling_results:
        logging.info("Doing predictions on the test set...")
        results = {}
        for target_name in targets_df.columns:
            model = modeling_results[target_name]["model"]
            y_val_pred = model.predict(features_df)
            y_val_prob = model.predict_proba(features_df)[:, 1]
            results[target_name] = {
                "test_recall": recall_score(targets_df[target_name], y_val_pred),
                "test_roc_auc": roc_auc_score(targets_df[target_name], y_val_prob),
                "test_precision": precision_score(targets_df[target_name], y_val_pred),
            }
        return results
