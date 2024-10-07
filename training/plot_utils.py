"""Module containing plotting functionalities"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .tensor_utils import *


def plot_losses(losses: list, nth: int = None):
    """Plot every nth loss value

    Args:
        losses (list): loss function values
        nth (int, optional): every nth value will be plotted. Defaults to None and every value will be plotted.
    """
    plt.plot(losses[0::nth])
    if nth:
        plt.title(f"RMSE over every {nth}th iteration")
    else:
        plt.title("RMSE over iterations")
    plt.ylabel("RMSE")
    plt.xlabel(f"Iteration X {nth if nth else 1}")


def plot_prediction_vs_truth(
    data: np.array,
    model: torch.nn.Module,
    forecast_horizon: int,
    prediction_window: int,
    stock_name: str,
) -> None:
    """Plot prediction from the model vs true values

    Args:
        data (np.array): true values
        model (torch.nn.Module): model to get the predictions
        forecast_horizon (int): number of days to predict
        prediction_windonw (int): number of days to use for prediction
    """
    data_as_tensor = torch.tensor(np.log(data), dtype=torch.float32).reshape(
        (-1, data.shape[-1], 1)
    )
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
    for _ in range(np.log(data[:, 0]).shape[0] - pred.shape[0]):
        pred = np.concatenate(([[np.nan]], pred))
    plt.plot(pred, label="predicted prices")
    plt.plot(np.log(data[:, 0]), label="true prices")
    plt.title(f"Prediction vs true prices for {stock_name} stock")
    plt.xlabel("days")
    plt.ylabel("log(prices)")
    plt.legend()
