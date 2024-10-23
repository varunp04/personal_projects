import pandas as pd
from typing import Dict, List


class gatherData:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder

    def load_data(self, file_name) -> pd.DataFrame:

        file_location = self.data_folder + file_name

        data_df = pd.read_csv(file_location)

        return data_df


class processData:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def select_specific_columns(self, data_df: pd.DataFrame) -> pd.DataFrame:

        column_list = (
            self.config.get("CATEGORICAL_ONE_HOT_ENCODE")
            + self.config.get("CATEGORICAL_LABEL_ENCODE")
            + self.config.get("NUMERICAL_COLUMNS")
            + self.config.get("DATE_COLUMNS")
        )

        column_list.append(self.config.get("TARGET_COLUMN"))

        data_df_selected_columns = data_df[column_list]

        return data_df_selected_columns

    def test_similar_index_for_nans(
        self, data_df: pd.DataFrame, column_name1: str, column_name2: str
    ) -> bool:

        list_index_column1 = data_df[data_df[column_name1].isna()].index
        list_index_column2 = data_df[data_df[column_name2].isna()].index

        return (list_index_column1 == list_index_column2).any()

    def replace_nans_with_str_val(
        self, data_df: pd.DataFrame, column_name: str, row_value: str
    ) -> pd.DataFrame:

        data_df[column_name] = data_df[column_name].fillna(row_value)

        return data_df

    def replace_nans_with_int_val(
        self, data_df: pd.DataFrame, column_name: str, row_value: int
    ) -> pd.DataFrame:

        data_df[column_name] = data_df[column_name].fillna(row_value)

        return data_df

    def remove_outliers(self, data_df: pd.DataFrame) -> pd.DataFrame:

        df_without_outliers = data_df[data_df["LotArea"] < 100000]

        df_without_outliers = df_without_outliers[
            df_without_outliers["TotalBsmtSF"] < 5000
        ]
        df_without_outliers = df_without_outliers[
            df_without_outliers["TotRmsAbvGrd"] < 14
        ]
        df_without_outliers = df_without_outliers[
            df_without_outliers["WoodDeckSF"] < 800
        ]
        df_without_outliers = df_without_outliers[
            df_without_outliers["OpenPorchSF"] < 500
        ]
        df_without_outliers = df_without_outliers[
            df_without_outliers["EnclosedPorch"] < 500
        ]

        return df_without_outliers

    def perform_data_processing(self, data_df: pd.DataFrame) -> pd.DataFrame:

        data_df_selected_columns = self.select_specific_columns(data_df=data_df)

        fintype_and_qual_match = self.test_similar_index_for_nans(
            data_df=data_df_selected_columns,
            column_name1="BsmtFinType1",
            column_name2="BsmtQual",
        )

        fintype_and_cond_match = self.test_similar_index_for_nans(
            data_df=data_df_selected_columns,
            column_name1="BsmtFinType1",
            column_name2="BsmtCond",
        )

        gar_type_and_qual_match = self.test_similar_index_for_nans(
            data_df=data_df_selected_columns,
            column_name1="GarageType",
            column_name2="GarageQual",
        )

        gar_type_and_cond_match = self.test_similar_index_for_nans(
            data_df=data_df_selected_columns,
            column_name1="GarageType",
            column_name2="GarageCond",
        )

        if (
            not fintype_and_cond_match
            or not fintype_and_qual_match
            or not gar_type_and_cond_match
            or not gar_type_and_qual_match
        ):
            raise Exception("Mismatch in index for Basement finish type or garage type")

        list_of_features_to_repalce_nans = [
            "BsmtFinType1",
            "GarageType",
            "MiscFeature",
            "BsmtQual",
            "BsmtCond",
            "GarageQual",
            "GarageCond",
        ]

        for column_name in list_of_features_to_repalce_nans:
            data_df_selected_columns = self.replace_nans_with_str_val(
                data_df=data_df_selected_columns,
                column_name=column_name,
                row_value="NA",
            )

        data_df_cleaned = data_df_selected_columns.dropna(subset=["Electrical"])

        data_df_cleaned["MSSubClass"] = data_df_cleaned["MSSubClass"].astype("str")

        data_df_cleaned[self.config.get("CATEGORICAL_ONE_HOT_ENCODE")] = (
            data_df_cleaned[self.config.get("CATEGORICAL_ONE_HOT_ENCODE")].astype(
                "category"
            )
        )

        data_df_cleaned[self.config.get("CATEGORICAL_LABEL_ENCODE")] = data_df_cleaned[
            self.config.get("CATEGORICAL_LABEL_ENCODE")
        ].astype("category")

        data_df_cleaned = data_df_cleaned.reset_index(drop=True)

        return data_df_cleaned
