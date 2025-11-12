import pandas as pd
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def split_data(data: pd.DataFrame):
    """
    Split the data into training and testing sets.

    Parameters:
        data (pd.DataFrame): Cleaned market data.

    Returns:
        tuple: Training set, testing set, and testing set with overlap.
    """

    # 60% train, 40% test
    train_size = int(len(data) * 0.6)

    # Split
    train = data[:train_size]
    test = data[train_size:]

    # Add one year overlap for backtesting to test
    overlap_size = 252  # Approx. number of trading days in a year
    test_plus = pd.concat([data[train_size - overlap_size:train_size], test])

    return train, test, test_plus

def test_data(tickers):
    """
    Download and prepare test data for the given tickers.

    Parameters:
        tickers (list): List of ticker symbols.

    Returns:
        tuple: Testing sets for both tickers.
    """

    # Download 15-year data
    data = yf.download(tickers, period="15y", auto_adjust=True, progress=False, group_by='ticker')

    # Prepare Close price DataFrames
    if isinstance(data.columns, pd.MultiIndex):
        df1 = data[tickers[0]][['Close']].copy()
        df2 = data[tickers[1]][['Close']].copy()
    else:
        df1 = data[['Close']].copy()
        df2 = data[['Close']].copy()

    # Split into train/test
    _, test1 = split_data(df1)
    _, test2 = split_data(df2)

    return test1, test2

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the DataFrame using mean and standard deviation scaling.

    Parameters:
        df (pd.DataFrame): DataFrame to normalize.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    
    norm_df = (df - df.mean()) / df.std()
    return norm_df
