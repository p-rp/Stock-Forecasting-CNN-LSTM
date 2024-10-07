""" Script containing methods to call the yahoo finance api. """
from typing import Any, List

import yfinance as yf


def get_stock_data(symbol: str, period: str) -> Any:

    """Get stock data by ticker.

    Args:
            symbol (str): Name of the symbol.

    Returns:
            Any: The stock history.
    """
    print(symbol)
    stock = yf.Ticker(symbol)

    try:
        stock_history = stock.history(period=period)
    except ValueError:
        print("get_stock_data: Invalid period value.")

    return stock_history


def get_batch_stock_data(symbols: List[str], period: str) -> Any:

    """Get stock data by ticker.

    Args:
            symbol (str): Name of the symbol.

    Returns:
            Any: The stock history.
    """
    print(symbols)
    try:
        batch_stock_history = yf.download(
            " ".join([s.upper() for s in symbols]), period=period, group_by="tickers"
        )
    except ValueError:
        print("get_batch_stock_data: Invalid stock list or period value.")

    return batch_stock_history
