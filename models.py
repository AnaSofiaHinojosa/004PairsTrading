from dataclasses import dataclass

@dataclass
class Operation:
    """
    Represents a single trading operation compatible with the backtest().
    """

    ticker: str              # Asset symbol (e.g., 'X' or 'Y')
    type: str                # "long" or "short"
    n_shares: int            # Number of shares/contracts
    entry: float             # Entry price
    exit: float              # Exit price (set when closing)
    time: str                # Timestamp or index of trade open