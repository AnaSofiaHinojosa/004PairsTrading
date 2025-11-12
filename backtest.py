from models import Operation
from cointegration import johansen_test
from kalman import KalmanFilter
import numpy as np


def get_portfolio_value(cash: float, 
                        long_ops: list[Operation], 
                        short_ops: list[Operation], 
                        x: float, 
                        y: float, 
                        x_price: float, 
                        y_price: float) -> float:

    """
    Calculate total portfolio value given cash and open positions.

    Parameters:
        cash (float): Current cash available.
        long_ops (list[Operation]): List of open long operations.
        short_ops (list[Operation]): List of open short operations.
        n_shares (int): Number of shares per operation.
        x (str): Ticker symbol for asset X.
        y (str): Ticker symbol for asset Y.
        x_price (float): Current price of asset X.
        y_price (float): Current price of asset Y.

    Returns:
        float: Total portfolio value.
    """

    val = cash

    # Add long positions value
    for pos in long_ops:
        if pos.ticker == x:
            pnl = x_price * pos.n_shares
            val += pnl
        if pos.ticker == y:
            pnl = y_price * pos.n_shares
            val += pnl    

    # Add short positions value
    for pos in short_ops:
        if pos.ticker == x:
            pnl = (pos.entry - x_price) * pos.n_shares
            val += pnl
        if pos.ticker == y:
            pnl = (pos.entry - y_price) * pos.n_shares
            val += pnl

    return val