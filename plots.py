import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as st
from models import Operation
import seaborn as sns


def plot_cointegrated_pair(df_1, df_2, ticker1, ticker2):
    """
    Plot the cointegrated pair of time series.

    Parameters:
        df_1 (pd.DataFrame): DataFrame containing the first time series.
        df_2 (pd.DataFrame): DataFrame containing the second time series.
        ticker1 (str): Ticker symbol for the first time series.
        ticker2 (str): Ticker symbol for the second time series.
    """
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_1.index, df_1[ticker1],
             label=ticker1, color='palevioletred')
    plt.plot(df_2.index, df_2[ticker2], label=ticker2, color='maroon')
    plt.title(f"Cointegrated Pair: {ticker1} & {ticker2}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_port_value(port_hist: list[float], dates: pd.Series) -> None:
    """
    Plot the portfolio value over time for the training set.

    Parameters:
        port_hist (list[float]): Portfolio values at each time step.
        dates (pd.Series): Corresponding datetime values.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, port_hist, color='maroon', alpha=0.7)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_spread(data) -> None:
    """
    Plot the spread over time.

    Parameters:
        data (pd.DataFrame): DataFrame containing the two time series.
    """

    data = data.copy()

    # Add constant for intercept
    X = st.add_constant(data.iloc[:, 0])
    y = data.iloc[:, 1]

    # Fit OLS model
    model = st.OLS(y, X).fit()
    residuals = model.resid
    mean_resid = residuals.mean()

    # Plot spread
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, color='palevioletred', label='Spread')
    plt.axhline(mean_resid, color='black', linestyle='--', label='Mean Spread')
    plt.title(f'Spread between {data.columns[0]} & {data.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_vecm_normalized(vecm_norm: list[float], theta: float, dates: pd.Series) -> None:
    """
    Plot the normalized VECM values over time.

    Parameters:
        vecm_norm (list[float]): Normalized VECM values at each time step.
        theta (float): Threshold value for trading signals.
        dates (pd.Series): Corresponding datetime values.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, vecm_norm, color='palevioletred')
    plt.axhline(theta, color='black', linestyle='--', label='Theta Threshold')
    plt.axhline(-theta, color='black', linestyle='--')
    plt.title("Normalized VECM Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalized VECM")
    plt.grid(linestyle=':', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_estimated(real1: list[float], estimated1: list[float], real2: list[float], estimated2: list[float], dates: pd.Series, type1: str, type2: str) -> None:
    """
    Plot the real and estimated values over time.

    Parameters:
        real1 (list[float]): Real values for the first type.
        estimated1 (list[float]): Estimated values for the first type.
        real2 (list[float]): Real values for the second type.
        estimated2 (list[float]): Estimated values for the second type.
        dates (pd.Series): Corresponding datetime values.
        type1 (str): Label for the first type.
        type2 (str): Label for the second type.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(dates, real1, label='Real ' + type1, color='palevioletred')
    axs[0].plot(dates, estimated1, label='Estimated ' +
                type1, color='maroon', alpha=0.7)
    axs[0].set_title(f"Real vs Estimated {type1} Over Time")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel(f"{type1} Values")
    axs[0].grid(linestyle=':', alpha=0.7)
    axs[0].legend()

    axs[1].plot(dates, real2, label='Real ' + type2, color='palevioletred')
    axs[1].plot(dates, estimated2, label='Estimated ' +
                type2, color='maroon', alpha=0.7)
    axs[1].set_title(f"Real vs Estimated {type2} Over Time")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel(f"{type2} Values")
    axs[1].grid(linestyle=':', alpha=0.7)
    axs[1].legend()

    plt.tight_layout()
    plt.show(block=True)


def plot_estimated_one(real, estimated, dates, type) -> None:
    """
    Plot the real and estimated values for a single type over time.

    Parameters:
        real (list[float]): Real values.
        estimated (list[float]): Estimated values.
        dates (pd.Series): Corresponding datetime values.
        type (str): Label for the type.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, real, label='Real ' + type, color='palevioletred')
    plt.plot(dates, estimated, label='Estimated ' +
             type, color='maroon', alpha=0.7)
    plt.title("Real " + type + " and Estimated " + type + " Over Time")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.grid(linestyle=':', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_kalman_weights(kalman_w1, dates) -> None:
    """
    Plot the Kalman filter weights over time.

    Parameters:
        kalman_w1 (list[float]): Kalman filter weights at each time step.
        dates (pd.Series): Corresponding datetime values.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, kalman_w1, label='Kalman Weight 1', color='palevioletred')
    plt.title("Hedge Ratio (Kalman Weight 1) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Kalman Weights")
    plt.grid(linestyle=':', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def hist_returns_distribution(positions: list[Operation]) -> None:
    """
    Plot the distribution of returns per trade.

    Parameters:
        positions (list[Operation]): List of trade positions.
    """

    returns = []

    # Calculate returns for each position
    for pos in positions:
        ret = pos.exit / pos.entry - 1
        if pos.type == 'SHORT':
            ret = -ret
        returns.append(ret)

    # Plot histogram
    plt.figure()
    sns.histplot(returns, color='palevioletred', alpha=0.3,
                 kde=True, bins=15, edgecolor=None)
    plt.title('Returns Distribution per Trade')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(linestyle=':', alpha=0.7)
    plt.show()
