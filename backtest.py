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


def backtest(data, original_eigenvector, theta) -> tuple:
    """
    Backtest trading strategy based on generated signals.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data for two assets.
        original_eigenvector (np.ndarray): Initial eigenvector from cointegration test.
        theta (float): Threshold for trading signals.

    Returns:
        tuple: Various results from the backtest including final cash, portfolio values,
               normalized VECM values, real and estimated prices, trade statistics, etc.
    """

    # Signals
    historic = data.copy()
    historic = historic.dropna()

    # Params
    COM = 0.125 / 100
    BORROW_RATE = 0.25 / 100

    cash = 1_000_000

    INTERVALS = 252  # daily intervals
    BORROW_RATE_DAILY = BORROW_RATE / INTERVALS

    # Initial conditions for Kalman Filters
    kalman_hedge = KalmanFilter(Q_mult=10, R_mult=0.001, P_mult=0.1)
    kalman_eigenvector = KalmanFilter(Q_mult=0.1, R_mult=0.001, P_mult=0.1)
    kalman_eigenvector.Q = np.array([[1, 0],
                                     [0, 0.001]])
    # Backtest logic
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []
    all_positions: list[Operation] = []

    portfolio_value = []

    buy = 0
    sell = 0
    hold = 0

    # Estimations and real value storage
    vecms_hat = []
    vecm_real = []
    vecms_norm = []

    estimated_p2 = []
    p2_real = []
    p1_real = []

    real_e2 = []
    real_e1 = []

    estimated_e2 = []
    estimated_e1 = []

    kalman1_w0 = []
    kalman1_w1 = []

    # Trade statistics
    trade_pnls = []
    total_commissions = 0
    total_borrow_costs = 0

    y = data.columns[0]
    x = data.columns[1]

    eigenvector_kalman = original_eigenvector

    for i, row in historic.iterrows():

        pos = data.index.get_loc(i)
        if pos < 252:
            portfolio_value.append(get_portfolio_value(
                cash, active_long_positions, active_short_positions, x, y, row[x], row[y]))
            continue

        p1 = row[y]
        p2 = row[x]

        # Update Kalman Filters

        # Filter 1 for hedge ratio
        y_kalman = p1
        x_kalman = p2

        kalman_hedge.update(x_kalman, y_kalman)
        w0, w1 = kalman_hedge.params
        hr = w1
        kalman1_w0.append(w0)
        kalman1_w1.append(w1)

        # Estimated P2
        p2_est = w1 * p1 + w0

        # Filter 2 for eigenvector
        x1_kalman = p1
        x2_kalman = p2

        if pos >= 252:
            try:
                _, _, _, _, eigenvector_kalman = johansen_test(
                    data.iloc[pos - 252:pos, :])
            except:
                pass

        estimated_p2.append(p2_est)
        p2_real.append(p2)
        p1_real.append(p1)

        e1_kalman, e2_kalman = eigenvector_kalman
        vecm = e1_kalman * x1_kalman + e2_kalman * x2_kalman
        vecm_real.append(vecm)
        real_e1.append(e1_kalman)
        real_e2.append(e2_kalman)

        kalman_eigenvector.update_vecm(x1_kalman, x2_kalman, vecm)
        e1_hat, e2_hat = kalman_eigenvector.params
        estimated_e1.append(e1_hat)
        estimated_e2.append(e2_hat)

        vecm_hat = e1_hat * x1_kalman + e2_hat * x2_kalman
        vecms_hat.append(vecm_hat)
        vecms_sample = vecms_hat[-252:]

        # Normalize VECM after having enough samples (252 observations)
        if len(vecms_sample) >= 252:
            mean_vecm = np.nanmean(vecms_sample)
            std_vecm = np.nanstd(vecms_sample)
            vecm_norm = (vecm_hat - mean_vecm) / \
                std_vecm if std_vecm != 0 else np.nan
        else:
            vecm_norm = np.nan

        vecms_norm.append(vecm_norm)

        # Check signals
        if abs(vecm_norm) < 0.05:
            # Close all positions

            # Close all long positions
            for position in active_long_positions.copy():
                # y
                if position.ticker == y:
                    start = position.entry * position.n_shares * (1 + COM)
                    end = row[y] * position.n_shares * (1 - COM)
                    pnl = end - start
                    commission = row[y] * position.n_shares * COM
                    cash += row[y] * position.n_shares * (1 - COM)
                    trade_pnls.append(pnl)
                    total_commissions += commission
                    # Update details
                    position.exit = row[y]

                # x
                if position.ticker == x:
                    start = position.entry * position.n_shares * (1 + COM)
                    end = row[x] * position.n_shares * (1 - COM)
                    pnl = end - start
                    commission = row[x] * position.n_shares * COM
                    cash += row[x] * position.n_shares * (1 - COM)
                    trade_pnls.append(pnl)
                    total_commissions += commission
                    # Update details
                    position.exit = row[x]

                # Remove position from active list
                active_long_positions.remove(position)

            # Close all short positions

            # Borrow cost
            for position in active_short_positions.copy():
                # y
                if position.ticker == y:
                    cover_cost = row[y] * position.n_shares
                    borrow_cost = cover_cost * BORROW_RATE_DAILY
                    cash -= borrow_cost
                    total_borrow_costs += borrow_cost

                # x
                if position.ticker == x:
                    cover_cost = row[x] * position.n_shares
                    borrow_cost = cover_cost * BORROW_RATE_DAILY
                    cash -= borrow_cost
                    total_borrow_costs += borrow_cost

            # Close short positions
            for position in active_short_positions.copy():
                # y
                if position.ticker == y:
                    pnl = (position.entry - row[y]) * position.n_shares
                    short_com = row[y] * position.n_shares * COM
                    cash += pnl - short_com
                    trade_pnls.append(pnl - short_com)
                    total_commissions += short_com
                    # Update details
                    position.x = row[y]

                # x
                if position.ticker == x:
                    pnl = (position.entry - row[x]) * position.n_shares
                    short_com = row[x] * position.n_shares * COM
                    cash += pnl - short_com
                    trade_pnls.append(pnl - short_com)
                    total_commissions += short_com
                    # Update details
                    position.exit = row[x]

                # Remove position from active list
                active_short_positions.remove(position)

        # Check signal

        if vecm_norm > theta and not active_long_positions and not active_short_positions:
            # Buy y
            available = cash * 0.4
            n_shares_long = available // (p1 * (1 + COM))
            if available > n_shares_long * p1 * (1 + COM) and n_shares_long > 0:
                # Do long
                position_value = p1 * n_shares_long * (1 + COM)
                cash -= position_value
                buy += 1
                operation = Operation(
                    ticker=y,
                    type="LONG",
                    n_shares=n_shares_long,
                    entry=p1,
                    exit=0.0,
                    time=i
                )
                active_long_positions.append(operation)
                all_positions.append(operation)

            # Short x
            n_shares_short = int(n_shares_long * hr)
            cost = p2 * n_shares_short * COM
            if cash > cost:
                # Do short
                cash -= cost
                sell += 1
                operation = Operation(
                    ticker=x,
                    type="SHORT",
                    n_shares=n_shares_short,
                    entry=p2,
                    exit=0.0,
                    time=i
                )
                active_short_positions.append(operation)
                all_positions.append(operation)
            else:
                hold += 1

        if vecm_norm < -theta and not active_long_positions and not active_short_positions:
            # Short y
            available = cash * 0.4
            n_shares_short = available // (p1 * (1 + COM))
            if available > n_shares_short * p1 * (1 + COM) and n_shares_short > 0:
                # Do short
                position_value = p1 * n_shares_short * COM
                cash -= position_value
                sell += 1
                operation = Operation(
                    ticker=y,
                    type="SHORT",
                    n_shares=n_shares_short,
                    entry=p1,
                    exit=0.0,
                    time=i
                )
                active_short_positions.append(operation)
                all_positions.append(operation)

            # Long x
            n_shares_long = int(n_shares_short * hr)
            position_value = p2 * n_shares_long * (1 + COM)
            if cash > position_value:
                # Do long
                cash -= position_value
                buy += 1
                operation = Operation(
                    ticker=x,
                    type="LONG",
                    n_shares=n_shares_long,
                    entry=p2,
                    exit=0.0,
                    time=i
                )
                active_long_positions.append(operation)
                all_positions.append(operation)

            else:
                hold += 1

        # Update portfolio value
        portfolio_value.append(get_portfolio_value(
            cash, active_long_positions, active_short_positions, x, y, row[x], row[y]))

    # Close remaining positions

    # Close all long positions
    for position in active_long_positions.copy():
        # y
        if position.ticker == y:
            start = position.entry * position.n_shares * (1 + COM)
            end = row[y] * position.n_shares * (1 - COM)
            pnl = end - start
            commission = row[y] * position.n_shares * COM
            cash += row[y] * position.n_shares * (1 - COM)
            trade_pnls.append(pnl)
            total_commissions += commission

            # Update details
            position.exit = row[y]

        # x
        if position.ticker == x:
            start = position.entry * position.n_shares * (1 + COM)
            end = row[x] * position.n_shares * (1 - COM)
            pnl = end - start
            commission = row[x] * position.n_shares * COM
            cash += row[x] * position.n_shares * (1 - COM)
            trade_pnls.append(pnl)
            total_commissions += commission
            # Update details
            position.exit = row[x]

    # Close all short positions

    # Close short positions
    for position in active_short_positions.copy():
        # y
        if position.ticker == y:
            pnl = (position.entry - row[y]) * position.n_shares
            short_com = row[y] * position.n_shares * COM
            cash += pnl - short_com
            trade_pnls.append(pnl - short_com)
            total_commissions += short_com
            # Update details
            position.exit = row[y]

        # x
        if position.ticker == x:
            pnl = (position.entry - row[x]) * position.n_shares
            short_com = row[x] * position.n_shares * COM
            cash += pnl - short_com
            trade_pnls.append(pnl - short_com)
            total_commissions += short_com
            # Update details
            position.exit = row[x]

    active_long_positions = []
    active_short_positions = []

    # Trade statistics
    num_trades = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    avg_win_loss = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    profit_factor = sum(wins) / abs(sum(losses)) if losses else np.nan

    trade_stats = {
        "Number of Trades": num_trades,
        "Number of Wins": len(wins),
        "Number of Losses": len(losses),
        "Win Rate": win_rate,
        "Average Win/Loss": avg_win_loss,
        "Profit Factor": profit_factor,
        "Total Commissions": total_commissions,
        "Total Borrow Costs": total_borrow_costs
    }

    return cash, portfolio_value, vecms_norm, p2_real, estimated_p2, real_e1, estimated_e1, real_e2, estimated_e2, p1_real, trade_stats, kalman1_w1, vecm_real, vecms_hat, all_positions
