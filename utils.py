"""Module containing general utility functions"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from prettytable import PrettyTable


def train_test_split(
    stock_df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the train and test split of the dataset

    Args:
        stock_df (pd.DataFrame): the dataset
        test_size (float, optional): the test split. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the train (index 0) and test  (index 1) sets
    """
    test_size = int(np.round(test_size * stock_df.shape[0]))
    train_size = stock_df.shape[0] - test_size
    return stock_df[:train_size], stock_df[train_size:]


def dict_to_str(title: str, dictionary: Dict) -> str:
    """Return the dictionary as a pretty string

    Args:
        dictionary (Dict): the dictionary

    Returns:
        str: the pretty string
    """
    table = PrettyTable()
    table.field_names = [c.replace("_", " ").title() for c in dictionary.keys()]
    table.add_row([dictionary.get(c, "") for c in dictionary.keys()])
    return table.get_string(title=title)
