# =============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# =============================================================================
import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =============================================================================
# 2. DEFINE THE TRADING STRATEGY (EMA CROSS)
# =============================================================================
class EmaCross(Strategy):
    n1 = 9  # Short-term EMA
    n2 = 30  # Long-term EMA

    def init(self):
        # Use exponential moving averages instead of simple ones
        self.ema1 = self.I(lambda x: pd.Series(x).ewm(span=self.n1, adjust=False).mean(), self.data.Close)
        self.ema2 = self.I(lambda x: pd.Series(x).ewm(span=self.n2, adjust=False).mean(), self.data.Close)

    def next(self):
        # Long-only behavior using EMA cross
        if crossover(self.ema1, self.ema2):
            if not self.position:
                self.buy(size=1)
        elif crossover(self.ema2, self.ema1):
            if self.position:
                self.position.close()


# --- Convenience Strategy class exposed for external orchestrators (FastAPI UI) ---
class EmaCrossStrategy(EmaCross):
    """Alias with 'Strategy' suffix so external code can find it by convention."""
    pass

# =============================================================================
# COMBINED STRATEGY: EMA CROSS + UT BOT ALERT (faithful to TradingView logic)
# =============================================================================
# =============================================================================
# COMBINED STRATEGY: EMA CROSS + UT BOT ALERT (faithful to TradingView logic)
# =============================================================================
class CombinedEmaUtBot(Strategy):
    # --- Parameters (match your TradingView inputs) ---
    ema_fast = 9
    ema_slow = 30
    key_value = 2        # corresponds to 'a' in Pine Script
    atr_period = 1       # corresponds to 'c' in Pine Script
    intraday_mode = True # 👈 toggle for day trading vs swing

    def init(self):
        # Convert to pandas Series to allow .shift()
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # --- EMA lines ---
        self.ema_fast_line = self.I(lambda x: pd.Series(x).ewm(span=self.ema_fast, adjust=False).mean(), self.data.Close)
        self.ema_slow_line = self.I(lambda x: pd.Series(x).ewm(span=self.ema_slow, adjust=False).mean(), self.data.Close)

        # --- ATR Calculation (safe version) ---
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        self.atr = self.I(lambda _: atr, self.data.Close)

        # --- UT Bot Trailing Stop (converted from PineScript) ---
        self.trailing_stop = self.I(self._ut_trailing_stop, close.values, atr.values, self.key_value)

    # --- UT Bot Trailing Stop Logic (faithful conversion) ---
    def _ut_trailing_stop(self, close, atr, key_value):
        nloss = key_value * atr
        ts = pd.Series(0.0, index=range(len(close)))

        for i in range(1, len(close)):
            prev = ts[i - 1] if i > 0 else 0

            if close[i] > prev and close[i - 1] > prev:
                ts[i] = max(prev, close[i] - nloss[i])
            elif close[i] < prev and close[i - 1] < prev:
                ts[i] = min(prev, close[i] + nloss[i])
            else:
                ts[i] = close[i] - nloss[i] if close[i] > prev else close[i] + nloss[i]

        return ts.values

    def next(self):
        close = self.data.Close[-1]
        trailing_stop = self.trailing_stop[-1]

        # UT Bot signals
        ut_buy = close > trailing_stop
        ut_sell = close < trailing_stop

        # EMA Cross logic
        ema_cross_up = crossover(self.ema_fast_line, self.ema_slow_line)
        ema_cross_down = crossover(self.ema_slow_line, self.ema_fast_line)

        # Combined Entry Rule: Both EMA cross + UT confirmation
        if ema_cross_up and ut_buy and not self.position:
            self.buy(size=1)

        # Exit Rule: Either EMA cross down or UT Bot sell
        elif (ema_cross_down or ut_sell) and self.position:
            self.position.close()

        # --- Optional intraday close rule ---
        if self.intraday_mode:
            t = self.data.index[-1]
            if hasattr(t, "time") and t.time().hour == 15 and t.time().minute >= 25:
                if self.position:
                    self.position.close()

# =============================================================================
# ORIGINAL SMA CROSS STRATEGY (COMMENTED OUT)
# =============================================================================
"""
class SmaCross(Strategy):
    n1 = 10  # Short-term moving average
    n2 = 30  # Long-term moving average

    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        # Long-only behavior:
        if crossover(self.sma1, self.sma2):
            if not self.position:
                self.buy(size=1)
        elif crossover(self.sma2, self.sma1):
            if self.position:
                self.position.close()
"""
# =============================================================================
# REMAINDER OF CODE UNCHANGED
# =============================================================================

class RsiStrategy(Strategy):
    period = 14
    rsi_buy = 30
    rsi_sell = 70

    def init(self):
        def _rsi(x, p):
            delta = pd.Series(x).diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
            rs = avg_gain / (avg_loss.replace(0, 1e-9))
            return 100 - 100 / (1 + rs)
        self.rsi = self.I(lambda x: _rsi(x, self.period), self.data.Close)

    def next(self):
        if self.rsi[-1] < self.rsi_buy and not self.position:
            self.buy(size=1)
        elif self.rsi[-1] > self.rsi_sell and self.position:
            self.position.close()


class MacdStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9

    def init(self):
        ser = pd.Series(self.data.Close)
        ema_fast = ser.ewm(span=self.fast, adjust=False).mean()
        ema_slow = ser.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal, adjust=False).mean()
        self.macd = self.I(lambda x: pd.Series(x).ewm(span=self.fast, adjust=False).mean() - pd.Series(x).ewm(span=self.slow, adjust=False).mean(), self.data.Close)
        self.macd_signal = self.I(lambda x: pd.Series(x).ewm(span=self.signal, adjust=False).mean(), self.macd)

    def next(self):
        if crossover(self.macd, self.macd_signal) and not self.position:
            self.buy(size=1)
        elif crossover(self.macd_signal, self.macd) and self.position:
            self.position.close()


# =============================================================================
# 3. PREPARE THE DATA
# =============================================================================
def normalize_yf_data(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        chosen = None
        ohlc_candidates = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        for lvl in range(data.columns.nlevels):
            vals = set(data.columns.get_level_values(lvl))
            if ohlc_candidates & vals:
                chosen = lvl
                break
        if chosen is not None:
            data.columns = data.columns.get_level_values(chosen)
        else:
            try:
                data.columns = data.columns.get_level_values(-1)
            except Exception:
                data.columns = [" ".join([str(p) for p in c if p is not None]) if isinstance(c, tuple) else str(c) for c in data.columns]

    data.columns = [c if isinstance(c, str) else str(c) for c in data.columns]
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    col_map = {c: c for c in data.columns}
    lower_to_col = {c.lower(): c for c in data.columns}
    for needed in ('open', 'high', 'low', 'close', 'volume'):
        if needed in lower_to_col and lower_to_col[needed] != needed.title():
            col_map[lower_to_col[needed]] = needed.title()
    if col_map and col_map != {c: c for c in data.columns}:
        data = data.rename(columns=col_map)
    return data


if __name__ == '__main__':
    symbol = 'RELIANCE.NS'
    print(f"Fetching historical stock data for {symbol}...")
    data = yf.download(symbol, start='2020-01-01', end=None)
    data = normalize_yf_data(data)

    if data.empty:
        print("No data fetched. Please check the ticker symbol or your internet connection.")
    else:
        print("Data fetched successfully!")
        print(f"Data range: {data.index.min()} to {data.index.max()}")

        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(set(data.columns)):
            print("Detected columns:", list(data.columns))
            raise ValueError("`data` must be a pandas.DataFrame with columns 'Open', 'High', 'Low', 'Close' (and optionally 'Volume').")

        bt = Backtest(data,
              CombinedEmaUtBot,
              cash=100000,
              commission=0.002)

        # For intraday testing
        CombinedEmaUtBot.intraday_mode = True

        stats = bt.run()
        print(stats)
        bt.plot()
