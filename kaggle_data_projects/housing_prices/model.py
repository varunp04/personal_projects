from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

model_Dict = {
    "decisiontree": DecisionTreeRegressor(),
    "randomforest": RandomForestRegressor(),
    "xgboost": XGBRegressor(),
    "lightgbm": LGBMRegressor(),
    "linearregression": LinearRegression(),
    "ridge": Ridge(),
}


class modelTraining:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def evaluate(
        self, y_actuals: np.ndarray, y_predictions: np.ndarray
    ) -> Tuple[float, float]:

        rmse = root_mean_squared_log_error(y_true=y_actuals, y_pred=y_predictions)
        mae = mean_absolute_error(y_true=y_actuals, y_pred=y_predictions)

        return rmse, mae

    def model_train(self, model_name: str, xtrain: np.ndarray, ytrain: np.ndarray):

        model = model_Dict[model_name]

        model.fit(xtrain, ytrain)

        return model

    def model_hyper_parameter_tuning(
        self, model_name: str, xtrain: np.ndarray, ytrain: np.ndarray
    ):

        model = model_Dict[model_name]
        params = self.config.get(model_name)
        regressor_search = GridSearchCV(
            model, params, cv=5, scoring="neg_root_mean_squared_log_error"
        )

        regressor_search.fit(xtrain, ytrain)
        best_params = regressor_search.best_params_
        best_estimator = regressor_search.best_estimator_

        return best_estimator, best_params

    def model_train_evaluate(
        self,
        modelname: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        tune: bool = False,
    ):

        if tune:
            trained_model, best_params = self.model_hyper_parameter_tuning(
                model_name=modelname, xtrain=x_train, ytrain=y_train
            )
            print(f"Best params: {best_params}")

            y_preds = trained_model.predict(x_test)
        else:

            trained_model = self.model_train(
                model_name=modelname, xtrain=x_train, ytrain=y_train
            )

            y_preds = trained_model.predict(x_test)

        rmse, mae = self.evaluate(y_actuals=y_test, y_predictions=y_preds)

        return rmse, mae
