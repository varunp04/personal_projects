from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from typing import Tuple, Dict
import numpy as np


class modelTraining:
    def __init__(self) -> None:
        pass

    def evaluate(self, actuals: np.ndarray, prediction: np.ndarray) -> Dict:

        metrics_dict = {}

        metrics_dict["accuracy"] = accuracy_score(y_true=actuals, y_pred=prediction)
        metrics_dict["f1"] = f1_score(y_true=actuals, y_pred=prediction, average="macro")
        metrics_dict["precision"] = precision_score(y_true=actuals, y_pred=prediction, average="macro")
        metrics_dict["recall"] = recall_score(y_true=actuals, y_pred=prediction, average="macro")

        return metrics_dict

    def split_data(
        self, data_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        x = data_df.drop(columns=["target"])
        y = data_df["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        return x_train, x_test, y_train, y_test

    def model_training(self, data_df: pd.DataFrame):

        x_train, x_test, y_train, y_test = self.split_data(data_df=data_df)

        dt_model = DecisionTreeClassifier()
        dt_model.fit(x_train, y_train)

        y_prediction = dt_model.predict(x_test)

        metrics_dict = self.evaluate(actuals=y_test, prediction=y_prediction)

        return metrics_dict
