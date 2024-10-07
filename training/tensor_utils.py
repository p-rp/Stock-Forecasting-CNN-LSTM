"""Module containing utility functions to manipulate tensors"""
import torch


def split_in_batch(
    data: torch.Tensor, forecast_horizon: int, prediction_window: int, drop_last: bool
) -> torch.Tensor:
    """Split the data in even chunks of size W = forecast_horizon + prediction_window

    Args:
            data (tensor): sequence of dimension (L, C_in, 1)
            forecast_horizon (int): number of days to predict
            prediction_windonw (int): number of days to use for prediction
            drop_last (bool): drop last chunk

    Returns:
            Tensor: batched data of dimension (N, C_in, W)
    """

    window = forecast_horizon + prediction_window
    batched_data = (
        torch.nn.utils.rnn.pad_sequence(torch.split(data, window, 0), batch_first=True)
        .transpose(1, 2)
        .squeeze(-1)
    )
    if drop_last:
        batched_data = batched_data[:-1, :, :]
    return batched_data


def sliding_window(
    data: torch.Tensor, forecast_horizon: int, prediction_window: int
) -> torch.Tensor:
    """Run a sliding window on the data of size W = forecast_horizon + prediction_window

    Args:
            data (tensor): sequence of dimension (L, C_in, 1)
            forecast_horizon (int): number of days to predict
            prediction_windonw (int): number of days to use for prediction

    Returns:
            Tensor: batched data of dimension (N, C_in, W)
    """

    window = forecast_horizon + prediction_window
    return data.unfold(0, window, forecast_horizon).squeeze(-2)
