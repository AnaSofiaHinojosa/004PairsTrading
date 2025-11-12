import numpy as np
import pandas as pd

class final_metrics():

    def __init__(self, portfolio_values: pd.Series):
        """
        Initialize the final_metrics class.

        Parameters:
            portfolio_values (pd.Series): Series of portfolio values over time.
        """

        self.portfolio_values = pd.Series(portfolio_values)

    def sharpe_ratio(self) -> float:
        """
        Calculate the Sharpe ratio of a portfolio.

        Returns:
            float: Sharpe ratio of the portfolio.
        """

        # Daily
        returns = self.portfolio_values.pct_change().dropna()
        mean = returns.mean()
        std = returns.std()

        # Annualized
        intervals = 252
        annual_rets = mean * intervals
        annual_std = std * np.sqrt(intervals)

        return annual_rets / annual_std if annual_std > 0 else 0


    def sortino_ratio(self) -> float:
        """
        Calculate the Sortino ratio of a portfolio.

        Returns:
            float: Sortino ratio of the portfolio.
        """

        # Daily
        returns = self.portfolio_values.pct_change().dropna()
        mean = returns.mean()
        downside = np.minimum(returns, 0).std()

        # Annualized
        intervals = 252
        annual_rets = mean * intervals
        annual_downside = downside * np.sqrt(intervals)

        return annual_rets / annual_downside if annual_downside > 0 else 0


    def max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown of a portfolio.

        Returns:
            float: Maximum drawdown of the portfolio.
        """

        rolling_max = self.portfolio_values.cummax()
        drawdowns = (self.portfolio_values - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        return abs(max_dd)


    def calmar_ratio(self) -> float:
        """
        Calculate the Calmar ratio of a portfolio.

        Returns:
            float: Calmar ratio of the portfolio.
        """

        # Daily
        returns = self.portfolio_values.pct_change().dropna()
        mean = returns.mean()

        # Annualized
        intervals = 252
        annual_rets = mean * intervals

        # Max Drawdown
        mdd = self.max_drawdown()

        return annual_rets / mdd if mdd > 0 else 0


    def evaluate_metrics(self) -> pd.DataFrame:
        """
        Evaluate key performance metrics of a portfolio.

        Returns:
            pd.DataFrame: DataFrame containing Sharpe ratio, Sortino ratio, Max drawdown, and Calmar ratio.
        """

        metrics = {
            'Sharpe ratio': self.sharpe_ratio(),
            'Sortino ratio': self.sortino_ratio(),
            'Max drawdown': self.max_drawdown(),
            'Calmar ratio': self.calmar_ratio()
        }

        return pd.DataFrame([metrics], index=['Value'])
    
    def summary(self):
        """
        Print a summary of the performance metrics.
        """

        print("=== PERFORMANCE METRICS ===")
        perf_metrics = self.evaluate_metrics()
        print(perf_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
        print("===========================")
