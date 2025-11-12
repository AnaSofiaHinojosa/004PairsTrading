import matplotlib.pyplot as plt
import pandas as pd


def plot_cointegrated_pair(df_1, df_2, ticker1, ticker2):
    norm_df1 = (df_1 - df_1.mean()) / df_1.std()
    norm_df2 = (df_2 - df_2.mean()) / df_2.std()
    plt.figure(figsize=(12,6))
    plt.plot(norm_df1.index, norm_df1[ticker1], label=ticker1, color='palevioletred')
    plt.plot(norm_df2.index, norm_df2[ticker2], label=ticker2, color='maroon')
    plt.title(f"Cointegrated Pair: {ticker1} & {ticker2}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()