import pandas as pd
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from plots import plot_cointegrated_pair
from utils import split_data


def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Perform Johansen cointegration test on the provided data.

    Parameters:
        data (pd.DataFrame): DataFrame containing the time series data of two assets.
        det_order (int): Deterministic trend order.
        k_ar_diff (int): Number of lagged differences in the VAR model.

    Returns:
        tuple: Trace statistic, critical values, and first eigenvector.
    """

    result = coint_johansen(data, det_order, k_ar_diff)
    trace_stat = result.lr1[0]
    crit_90, crit_95, crit_99 = result.cvt[0]
    first_eigenvector = result.evec[:, 0]

    return trace_stat, crit_90, crit_95, crit_99, first_eigenvector


def calculate_rolling_correlation(df, window=252):
    """
    Calculate the rolling correlation between two time series.

    Parameters:
        df (pd.DataFrame): DataFrame containing two columns of time series data.
        window (int): Rolling window size.

    Returns:
        float: Mean of the rolling correlations.
    """

    rolling_corr = df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])

    # Return the mean of the rolling correlations
    return rolling_corr.mean()


def run_ols_adf(df):
    """
    Perform OLS regression and ADF test on the residuals.

    Parameters:
        df (pd.DataFrame): DataFrame containing two columns of time series data.

    Returns:
        tuple: Residuals from OLS regression and p-value from ADF test.
    """

    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    residuals = model.resid
    adf_result = adfuller(residuals)
    p_value = adf_result[1]

    return residuals, p_value

def individual_adf_test(df):
    """
    Perform ADF test on each individual time series.

    Parameters:
        df (pd.DataFrame): DataFrame containing two columns of time series data.

    Returns:
        tuple: p-values from ADF tests for both series.
    """
 
    adf_result_1 = adfuller(df.iloc[:, 0])
    p_value_1 = adf_result_1[1]

    adf_result_2 = adfuller(df.iloc[:, 1])
    p_value_2 = adf_result_2[1]

    return p_value_1, p_value_2


def cointegration_analysis():
    """
    Perform cointegration analysis on predefined stock pairs within sectors.

    Returns:
        tuple: DataFrame of filtered results and DataFrame of all results.
    """

    # Define tickers per sector
    sector_tickers = {
    "Midstream_Energy": [
        "TRGP", "OKE", "KMI", "ET", "WMB"
    ],

    "Industrial_Automation": [
        "ROK", "SIEGY", "ETN", "EMR"
    ],

    "Banks": [
        "JPM", "C", "WFC", "USB", "MS", "GS"
    ],

    "Consumer_Staples": [
        "PEP", "PG", "K", "SJM", "HSY", "GIS", "KO"
    ],
    
    "Healthcare_Equipment": [
        "MDT", "SYK", "BSX", "DHR", "ZBH"
    ]
}
    
    # Generate all possible pairs within each sector
    sector_pairs = {sector: list(combinations(tickers, 2))
                    for sector, tickers in sector_tickers.items()}
    results = []

    for sector, pairs in sector_pairs.items():
        print(f"Testing {sector} sector...")
        for t1, t2 in pairs:
            try:
                # Download 15-year data
                data = yf.download(
                    [t1, t2], period="15y", group_by='ticker', auto_adjust=True, progress=False)
                df_close = pd.DataFrame()
                for t in [t1, t2]:
                    if isinstance(data.columns, pd.MultiIndex):
                        df_close[t] = data[t]["Close"]
                    else:
                        df_close[t] = data["Close"]

                if df_close.shape[0] < 250:
                    continue

                df_close = df_close.dropna()

                # Split train/test
                train_df, _, _ = split_data(df_close)

                # Rolling correlation filter
                avg_corr = calculate_rolling_correlation(train_df)

                # OLS + ADF test
                _, adf_p = run_ols_adf(train_df)

                # Individual ADF tests
                adf_p_1, adf_p_2 = individual_adf_test(train_df)

                # Johansen test
                trace_stat, _, c95, _, first_eigenvector = johansen_test(
                    train_df)
                cointegrated_johansen = trace_stat > c95

                # Append only pairs passing correlation & ADF
                results.append({
                    "Sector": sector,
                    "Ticker1": t1,
                    "Ticker2": t2,
                    "RollingCorr": avg_corr,
                    "ADF_pvalue": adf_p,
                    "TraceStat": trace_stat,
                    "Crit_95": c95,
                    "Cointegrated_Johansen": cointegrated_johansen,
                    "Strength": trace_stat / c95,
                    "Eigenvector": first_eigenvector,
                    "ADF_pvalue_1": adf_p_1,
                    "ADF_pvalue_2": adf_p_2
                })

            except Exception as e:
                print(f"{t1}-{t2} failed: {e}")

    df_results = pd.DataFrame(results)
    df_filtered = df_results[(df_results['RollingCorr'] > 0.6) & (
        df_results['ADF_pvalue'] < 0.05) & (df_results['Cointegrated_Johansen'] == True)]
    df_sorted = df_filtered.sort_values(
        by="Strength", ascending=False).reset_index(drop=True)

    return df_sorted, df_results


def get_test_results(df_results: pd.DataFrame) -> None:
    """
    Export cointegration test results to an Excel file.

    Parameters:
        df_results (pd.DataFrame): DataFrame containing cointegration test results.
    """

    # Export DataFrame to xlsx
    df_results.to_excel("cointegration_results.xlsx", index=False)


def choose_pair(df_results: pd.DataFrame) -> tuple:
    """
    Choose the best cointegrated pair based on test results.

    Parameters:
        df_results (pd.DataFrame): DataFrame containing cointegration test results.

    Returns:
        tuple: Full DataFrame of the pair, training set, testing set, extended testing set,
               list of tickers, and first eigenvector.
    """

    # Select strongest pair passing Johansen test
    best_idx = df_results[df_results['Cointegrated_Johansen']
                          ]["Strength"].idxmax()
    best_pair = df_results.loc[best_idx]
    tickers = [best_pair['Ticker1'], best_pair['Ticker2']]
    first_eigenvector = best_pair["Eigenvector"]

    # Download full 15-year data
    data = yf.download(tickers, period="15y", auto_adjust=True,
                       progress=False, group_by='ticker')

    # Prepare close price DataFrame
    df_pair = pd.DataFrame()
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            df_pair[t] = data[t]["Close"]
        else:
            df_pair[t] = data["Close"]

    df_pair = df_pair.dropna()

    # Split into train/test
    train_df, test_df, test_plus_df = split_data(df_pair)

    # Plot training portion
    plot_cointegrated_pair(train_df[[tickers[0]]], train_df[[
                           tickers[1]]], tickers[0], tickers[1])

    return df_pair, train_df, test_df, test_plus_df, tickers, first_eigenvector
