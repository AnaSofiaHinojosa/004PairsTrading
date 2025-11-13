from cointegration import cointegration_analysis, choose_pair
from plots import *
from backtest import backtest
from metrics import final_metrics


def trials():
    # Possible theta values to test
    theta = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    sortinos = []

    # Perform cointegration analysis
    df_results, _ = cointegration_analysis()

    # Choose the strongest cointegrated pair and get train/test
    _, _, _, test_plus_df, _, eigenvector = choose_pair(
        df_results)

    for t in theta:
        # Backtest the strategy
        _, port_value, _, _, _, _, _, _, _, _, _, _, _, _, _ = backtest(
            test_plus_df, eigenvector, t)

        # Metrics evaluation
        print(f"\nTheta: {t}")
        metrics = final_metrics(port_value)
        metrics.summary()
        sortino = metrics.sortino_ratio()
        sortinos.append(sortino)

    plot_theta_sortino(theta, sortinos)

if __name__ == "__main__":
    trials()
