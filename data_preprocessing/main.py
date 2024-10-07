"""Driver module to run the data collection scripts"""
import argparse
import os.path
import ssl
import urllib.request

import pandas as pd
from yahoo import *


def fetch_symbols(directory: str) -> pd.Series:
    """Fetches the list of S&P 500 from Wikipedia and saves.

    Returns:
        pd.Series: Stock symbols of S&P 500 companies.
    """
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df_to_csv(df, filename="stock-symbols.csv", directory=directory, columns=["Symbol"])

    return df["Symbol"]


def read_symbols(path: str) -> pd.DataFrame:
    """Fetches the list of S&P 500 from Wikipedia and saves.

    Returns:
        pd.Series: Stock symbols of S&P 500 companies.
    """
    df = pd.read_csv(path, sep=" ", header=None)

    return df


def df_to_csv(
    df: pd.DataFrame,
    filename: str,
    directory: str = None,
    columns: List[str] = None,
    header: bool = False,
) -> None:
    """Writes dataframe into text file (fmt: .csv).

    Args:
        df (pd.DataFrame): The data frame to save.
        filename (str): Name of the file in directory.
        directory (str, optional): Name of the directory in project. Defaults to None.
        columns (List[str], optional): List of columns in to_csv function. Defaults to None.
    """
    path = filename if not directory else "{}/{}".format(directory, filename)
    df.to_csv(path, columns=columns, index=False, header=header)


def main() -> None:
    """Main method containing the logic invocation."""
    parser = argparse.ArgumentParser(description="Data collection and preprocessing")

    # Fetch a fresh S&P500 stock list from Wikipedia
    parser.add_argument(
        "-f",
        "--fresh",
        action="store_true",
        help="generate a new csv file with all stock symbols",
    )

    parser.add_argument("-d", action="store", help="the directory to store results in")

    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context

    # Get S&P 500 stock symbols.
    if args.fresh or not os.path.isfile("../{}/stock-symbols.csv".format(args.d)):
        _symbols = fetch_symbols(directory=args.d)

    symbols = read_symbols("./{}/stock-symbols.csv".format(args.d))

    # Get historical data (past 10 yrs.) for each one and save to directory.
    for _, s in symbols.iterrows():
        res = get_stock_data(s[0], period="10y")
        df_to_csv(
            res,
            filename="{}-info.csv".format(s[0]),
            directory=args.d,
            header=True,
        )


if __name__ == "__main__":
    main()
