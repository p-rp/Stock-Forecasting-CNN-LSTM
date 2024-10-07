"""Module containing utility functions for cross-validation and testing"""
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.plot_utils import *
from training.tensor_utils import *
from training.training import *

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)
import random

random.seed(1)


def predict_trend(
    data: pd.DataFrame,
    model: torch.nn.Module,
    forecast_horizon: int,
    prediction_window: int,
) -> float:
    """Predict if the next day's price will go up or down and return the accuracy over the whole dataset

    Args:
        data (pd.DataFrame): dataset
        model (torch.nn.Module): pretrained model
        forecast_horizon (int): number of days to predict
        prediction_window (int): number of days to use for the prediction

    Returns:
        float: prediction accuracy
    """
    preds = get_prediction(data, model, forecast_horizon, prediction_window)
    prev = preds[0]
    trend = []
    for pred in preds[1:]:
        if pred > prev:
            trend.append(1)
        else:
            trend.append(0)
        prev = pred
    open_prices = np.log(data.loc[:, ["Open"]].to_numpy()).reshape((-1, 1))
    diff = open_prices.shape[0] - preds.shape[0]
    prev2 = open_prices[diff]
    true_trend = []
    for true in open_prices[diff + 1 :]:
        if true > prev2:
            true_trend.append(1)
        else:
            true_trend.append(0)
        prev2 = true
    cnt = 0
    for i in range(len(true_trend)):
        if true_trend[i] == trend[i]:
            cnt += 1
    return cnt / len(true_trend)


def get_prediction(
    data: pd.DataFrame,
    model: torch.nn.Module,
    forecast_horizon: int,
    prediction_window: int,
) -> np.array:
    """Get prediction from the model

    Args:
        data (pd.DataFrame): dataset
        model (torch.nn.Module): pretrained model
        forecast_horizon (int): number of days to predict
        prediction_window (int): number of days used for the prediction

    Returns:
        np.array: predictions made by the model
    """
    data_to_transform = data.loc[:, ["Open"]].to_numpy()
    data_as_tensor = torch.tensor(
        np.log(data_to_transform), dtype=torch.float32
    ).reshape((-1, data_to_transform.shape[-1], 1))

    pred = (
        model(
            sliding_window(data_as_tensor, forecast_horizon, prediction_window)[
                :, :, :prediction_window
            ]
        )[-1]
        .detach()
        .numpy()
        .reshape((-1, 1))
    )

    return pred


def testing_accuracy(
    test: pd.DataFrame,
    model: torch.nn.Module,
    forecast_horizon: int,
    prediction_window: int,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Compute the MSE and RMSE

    Args:
        test (pd.DataFrame): testing dataset
        model (torch.nn.Module): pretrained model
        forecast_horizon (int): number of days to predict
        prediction_window (int): number of days to use in the prediction
        verbose (bool, optional): Print the errors. Defaults to True.

    Returns:
        tuple[float, float]: _description_
    """
    pred = get_prediction(test, model, forecast_horizon, prediction_window)
    data_to_transform = test.loc[:, ["Open"]].to_numpy()
    data_as_tensor = torch.tensor(
        np.log(data_to_transform), dtype=torch.float32
    ).reshape((-1, data_to_transform.shape[-1], 1))
    s = sliding_window(data_as_tensor, forecast_horizon, prediction_window)[
        :, :, -forecast_horizon:
    ].reshape((-1, 1))

    mse = np.square(np.subtract(pred, s)).mean().detach().numpy().item()
    rmse = math.sqrt(mse)

    if verbose:
        print(
            "Mean Square Error (MSE): {:.2f}\nRoot Mean Square Error (RMSE): {:.2f}".format(
                mse, rmse
            )
        )

    return mse, rmse


def cross_validate(
    data: pd.DataFrame, folds: int, forecast_horizon: int, prediction_window: int
) -> float:
    """Performs K-fold cross validation with the given set of hyperparameters and return the average error

    Args:
        data (pd.DataFrame): dataset
        folds (int): number of folds for CV
        forecast_horizon (int): number of days to predict
        prediction_window (int): number of days to use in the prediction

    Returns:
        float: average RMSE
    """
    kf = KFold(n_splits=folds, shuffle=False)
    errors = []

    for _, (training_index, testing_index) in enumerate(kf.split(data)):
        model, _, _, _ = train_with_hyperparameters(
            data.iloc[training_index],
            0.01,
            0.9,
            ["Open"],
            forecast_horizon,
            prediction_window,
            128,
        )
        data2 = data.iloc[testing_index].loc[:, ["Open"]].to_numpy()
        data_as_tensor = torch.tensor(np.log(data2), dtype=torch.float32).reshape(
            (-1, data2.shape[-1], 1)
        )
        pred = get_prediction(
            data.iloc[testing_index], model, forecast_horizon, prediction_window
        )
        s = sliding_window(data_as_tensor, forecast_horizon, prediction_window)[
            :, :, -forecast_horizon:
        ].reshape((-1, 1))

        MSE = np.square(np.subtract(pred, s)).mean()
        RMSE = math.sqrt(MSE)
        errors.append(RMSE)

    avg_error = np.mean(errors)
    print(f"CHECK OUT THIS AVERAGE ERROR: {avg_error}")
    return avg_error


def hyperparam_search(data: pd.DataFrame, folds: int) -> List[dict]:
    """Perform grid search on forecast horizon and prediction window

    Args:
        data (pd.DataFrame): dataset
        folds (int): number of folds for CV

    Returns:
        list[dict]: list of dictionaries containing the hyperparameters and the error value
    """
    forecast_horizon = [1, 2, 5, 10, 15]
    prediction_window = [5, 10, 20, 40]
    result = []
    for fh in forecast_horizon:
        for pw in prediction_window:
            print(f"FOR FORECAST HORIZON {fh} AND PREDICTION WINDOW {pw}")
            avg_error = cross_validate(data, folds, fh, pw)
            result.append(
                {"forecast_horizon": fh, "prediction_window": pw, "error": avg_error}
            )
    return result
