import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import cross_validate, KFold

N_COMPONENTS = 0.95
N_FOLDS = 10
SEED = 123

np.random.seed(SEED)
random.seed(SEED)

class CustomLogisticRegression(LogisticRegression):
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        return super().fit(X, y)

def custom_scorer(estimator, X, y):
    if not isinstance(estimator, CustomLogisticRegression):
        raise ValueError("Estimator must be an instance of CustomLogisticRegression ...")

    y_val_pred = estimator.predict(X)
    val_precision = precision_score(y_true=y, y_pred=y_val_pred)
    val_accuracy = accuracy_score(y_true=y, y_pred=y_val_pred)

    y_train_pred = estimator.predict(estimator.X_train)
    train_precision = precision_score(y_true=estimator.y_train, y_pred=y_train_pred)
    train_accuracy = accuracy_score(y_true=estimator.y_train, y_pred=y_train_pred)

    return {
        "val_precision": val_precision,
        "val_accuracy": val_accuracy,
        "train_precision": train_precision,
        "train_accuracy": train_accuracy,
    }

def pca_logistic_classifier(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "test"] = "train",
        prediction_models: dict[str, object] = None,
) -> dict[str, object] | None:
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    if operation_mode == "train":
        logging.info("Removing constant features...")
        features_copy_df = features_df.copy(deep=True)
        features_df = features_copy_df.loc[:, ~(features_copy_df.nunique() == 1)]

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=N_COMPONENTS).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Creating cross-validation sets...")
        cv = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
        logging.info("Cross-validation sets created successfully.")

        logging.info("Training a logistic regression model on PCA components...")
        results = {}
        for target_name in targets_df.columns:
            model = CustomLogisticRegression(random_state=SEED)
            scores = cross_validate(
                estimator=model,
                X=pca_features_df,
                y=targets_df[target_name],
                scoring=custom_scorer,
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            results[target_name] = {
                "model": model.fit(X=pca_features_df, y=targets_df[target_name]),
                "val_accuracy": np.mean(scores["test_val_accuracy"]),
                "val_precision": np.mean(scores["test_val_precision"]),
                "train_accuracy": np.mean(scores["train_train_accuracy"]),
                "train_precision": np.mean(scores["train_train_precision"]),
            }
        logging.info("Training completed successfully.")

        results["n_components"] = pca_features_df.shape[1]
        return results

    if operation_mode == "test" and prediction_models is not None:
        if prediction_models["n_components"] is None:
            logging.error("PCA components count is not provided in the prediction models...")
            return None

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=prediction_models["n_components"]).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Performing predictions on test set...")
        results = {}
        for target_name in targets_df.columns:
            model = prediction_models[target_name]["model"]
            y_pred = model.predict(pca_features_df)
            results[target_name] = {
                "precision": precision_score(y_true=targets_df[target_name], y_pred=y_pred),
                "accuracy": accuracy_score(y_true=targets_df[target_name], y_pred=y_pred),
                "test_features": pca_features_df,
            }
        logging.info("Prediction completed successfully.")
        return results
