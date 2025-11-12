from cointegration import cointegration_analysis, choose_pair, get_test_results
from plots import *
from backtest import backtest
from metrics import final_metrics
import pandas as pd

def main():
    # Perform cointegration analysis
    df_results, df_all_results = cointegration_analysis()

    # Export results to Excel
    # get_test_results(df_all_results)

    print("Cointegration Analysis Results:")
    print(df_results)

    # Choose the strongest cointegrated pair and get train/test
    full_df, train_df, test_df, test_plus_df, _, eigenvector = choose_pair(df_results)
    print("Selected Pair:")
    print(full_df.columns.tolist())

if __name__ == "__main__":
    main()    