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

    # Backtest the strategy
    theta =  0.84
    cash, port_value, vecms_norm, p2_real, estimated_p2, real_e1, estimated_e1, real_e2, estimated_e2, _, trade_stats, kalman1_w1, vecm_real, vecm_hat, all_positions = backtest(test_plus_df, eigenvector, theta)

    # Metrics evaluation
    metrics = final_metrics(port_value)
    metrics.summary()

    # Trade statistics
    print("Trade Statistics:")
    for key, value in trade_stats.items():
        print(f"  {key}: {value}")
    print("===========================")

    # Portfolio final value and cash
    final_value = port_value[-1]    
    print(f"\nFinal Portfolio Value: ${final_value:,.2f}")
    print(f"Final Cash: ${cash:,.2f}")

if __name__ == "__main__":
    main()