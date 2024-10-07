"""Module containing utility functions to train the model"""
from typing import List

import numpy as np
import pandas as pd
import torch

from .model.net import CNN_LSTM
from .tensor_utils import *

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)
import random

random.seed(1)


def batch_train(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: torch.nn,
    optimizer: torch.optim,
    forecast_horizon: int,
    prediction_window: int,
) -> float:
    """Forward and backward pass on a batch
    Args:
            model (torch.nn.Module): model being trained
            data (torch.Tensor): batch data
            loss_function (torch.nn): loss function being optimized
            optimizer (torch.nn): optimizer to minimize the loss
            forecast_horizon (int): number of days being predicted
            prediction_window (int): number of days used for prediction
    Returns:
            float: loss value of that batch
    """
    optimizer.zero_grad()
    X = data[:, :, :prediction_window]
    y = data[:, 0, prediction_window:]
    pred = model(X)[-1][0, :, :]
    loss = loss_function(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: torch.nn,
    optimizer: torch.optim,
    batch_size: int,
    num_epoch: int,
    verbose: bool,
    forecast_horizon: int,
    prediction_window: int,
):
    """Train the model on specified number of epochs
    Args:
        model (torch.nn.Module): model being trained
        data (torch.Tensor): data
        loss_function (torch.nn): loss function being optimized
        optimizer (torch.nn): optimizer to minimize the loss
        batch_size (int): batch size for training
        num_epoch (int): number of epochs to perform
        verbose (bool): if True, prints loss value at each epoch
        forecast_horizon (int): number of days being predicted
        prediction_window (int): number of days used for prediction
    Returns:
        float: loss value of that batch
    """
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=True
    )
    losses = []
    for epoch in range(num_epoch):
        for batch in dataloader:
            batched_data = sliding_window(batch[0], forecast_horizon, prediction_window)
            loss = batch_train(
                model,
                batched_data,
                loss_function,
                optimizer,
                forecast_horizon,
                prediction_window,
            )
            losses.append(loss)
        if verbose:
            print(f"epoch {epoch}, loss: {loss}")

    return losses


def train_with_hyperparameters(
    data: pd.DataFrame,
    learning_rate: float,
    momentum: float,
    features: List[str],
    forecast_horizon: int,
    prediction_window: int,
    batch_size: int,
    model: torch.nn.Module = None,
    loss: torch.nn = None,
    optimizer: torch.optim = None,
    verbose: bool = False,
) -> tuple:
    """Train the model parameterized by the passed hyperparameters
    Args:
        data (torch.Tensor): data
        learning_rate (float): learning rate for optimizer
        momentum (float): momentum for optimizer
        features (list): features to use to predict the open price
        forecast_horizon (int): number of days being predicted
        prediction_window (int): number of days used for prediction
        batch_size (int): batch size for training
        model (torch.nn.Module): pre-trained model
        loss (torch.nn): loss function associated with pre-trained model
        optimizer (torch.nn): optimizer associated with pre-trained model
        verbose (bool): flag to output loss values at each epoch
    Returns:
        tuple: model, loss, optimizer, loss values
    """
    try:
        price_data = data.loc[:, features].to_numpy()
    except KeyError:
        raise ValueError(f"{features} is not in passed data")

    data = torch.tensor(np.log(price_data), dtype=torch.float32).reshape(
        (-1, len(features), 1)
    )
    if not model:
        model = CNN_LSTM(
            kernel_size=2,
            n_filters1=32,
            n_filters2=64,
            pool_size=2,
            hidden_size=100,
            num_features=len(features),
            output_size=forecast_horizon,
        )
    if not loss:
        loss = torch.nn.MSELoss()
    if not optimizer:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )
    losses = train(
        model,
        data,
        loss,
        optimizer,
        batch_size,
        50,
        verbose,
        forecast_horizon,
        prediction_window,
    )
    return model, loss, optimizer, losses
