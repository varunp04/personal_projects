import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
import utils
import numpy as np
from sklearn.model_selection import train_test_split


class EngineerData:

    def __init__(self, config: Dict) -> None:
        self.config = config

    def select_high_corr_features(self, data_df: pd.DataFrame) -> pd.DataFrame:

        all_numerical_features = self.config.get("NUMERICAL_COLUMNS") + self.config.get(
            "DATE_COLUMNS"
        )

        for column in self.config.get("HIGH_CORRELATED_COLUMN"):
            all_numerical_features.remove(column)

        data_df = data_df.drop(columns=all_numerical_features)

        return data_df

    def one_hot_encoding(
        self, data_df: pd.DataFrame
    ) -> Tuple[OneHotEncoder, pd.DataFrame]:

        one_hot_encode = OneHotEncoder(
            sparse_output=False, handle_unknown="infrequent_if_exist"
        ).set_output(transform="pandas")
        encoded_data = one_hot_encode.fit_transform(data_df)

        return one_hot_encode, encoded_data

    def label_encoding(self, data_df: pd.DataFrame, column: str) -> pd.DataFrame:

        label = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}

        data_df[column] = data_df[column].replace(label)

        return data_df

    def perform_feature_eng(self, data_df: pd.DataFrame) -> pd.DataFrame:

        data_high_corr = self.select_high_corr_features(data_df=data_df)

        ohencoder, encoded_data = self.one_hot_encoding(
            data_df=data_high_corr[self.config.get("CATEGORICAL_ONE_HOT_ENCODE")]
        )

        data_high_corr = data_high_corr.drop(
            columns=self.config.get("CATEGORICAL_ONE_HOT_ENCODE")
        )

        data_one_hot_encoded = pd.concat([data_high_corr, encoded_data], axis=1)

        utils.save_as_pickle(
            path=self.config.get("ARTIFACT_FOLDER"),
            artifact_name="ohencoder.pkl",
            artifact=ohencoder,
        )

        utils.save_as_pickle(
            path=self.config.get("ARTIFACT_FOLDER"),
            artifact_name="ohencoded_features.yaml",
            artifact=encoded_data.columns.tolist(),
        )

        data_all_encoded = self.label_encoding(
            data_df=data_one_hot_encoded, column="GarageQual"
        )

        data_all_encoded[self.config.get("TARGET_COLUMN")] = np.log1p(
            data_all_encoded[self.config.get("TARGET_COLUMN")]
        )

        return data_all_encoded

    def split_data(
        self, data_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        x = data_df.drop(columns=[self.config.get("TARGET_COLUMN")])
        y = data_df[self.config.get("TARGET_COLUMN")]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        return x_train, x_test, y_train, y_test
