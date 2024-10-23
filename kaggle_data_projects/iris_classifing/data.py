from sklearn.datasets import load_iris
import pandas as pd


class gatherData:
    def __init__(self) -> None:
        pass

    def load_data(self) -> pd.DataFrame:
        data = load_iris(as_frame=True)

        master_data = data["data"].copy()
        master_data["target"] = data["target"]

        return master_data
