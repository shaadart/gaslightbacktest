# =============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# =============================================================================
import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =============================================================================
# 2. DEFINE THE TRADING STRATEGY (The "Blueprint")
# =============================================================================
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


# --- Convenience Strategy classes exposed for external orchestrators (FastAPI UI) ---
class SmaCrossStrategy(SmaCross):
    """Alias with 'Strategy' suffix so external code can find it by convention."""
    pass


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
        # register indicators for plotting
        self.macd = self.I(lambda x: pd.Series(x).ewm(span=self.fast, adjust=False).mean() - pd.Series(x).ewm(span=self.slow, adjust=False).mean(), self.data.Close)
        self.macd_signal = self.I(lambda x: pd.Series(x).ewm(span=self.signal, adjust=False).mean(), self.macd)

    def next(self):
        if crossover(self.macd, self.macd_signal) and not self.position:
            self.buy(size=1)
        elif crossover(self.macd_signal, self.macd) and self.position:
            self.position.close()


class CombinedStrategy(Strategy):
    """Enter when at least two signals agree (SMA crossover, RSI oversold, MACD crossover)."""
    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(10).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(30).mean(), self.data.Close)
        def _rsi(x, p=14):
            delta = pd.Series(x).diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
            rs = avg_gain / (avg_loss.replace(0, 1e-9))
            return 100 - 100 / (1 + rs)
        self.rsi = self.I(lambda x: _rsi(x, 14), self.data.Close)
        self.macd = self.I(lambda x: pd.Series(x).ewm(span=12, adjust=False).mean() - pd.Series(x).ewm(span=26, adjust=False).mean(), self.data.Close)
        self.macd_signal = self.I(lambda x: pd.Series(x).ewm(span=9, adjust=False).mean(), self.macd)

    def next(self):
        signals = 0
        if crossover(self.sma1, self.sma2): signals += 1
        if self.rsi[-1] < 35: signals += 1
        if crossover(self.macd, self.macd_signal): signals += 1

        if signals >= 2 and not self.position:
            self.buy(size=1)
        elif signals < 2 and self.position:
            self.position.close()


# =============================================================================
# 3. PREPARE THE DATA
# =============================================================================
# Helper: normalize yfinance DataFrame to single-level OHLC columns
def normalize_yf_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance DataFrame so columns are single-level strings like Open/High/Low/Close/Volume."""
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
    print("Fetching historical stock data for Reliance Industries...")
    data = yf.download('RELIANCE.NS', start='2020-01-01', end=None)
    data = normalize_yf_data(data)

    if data.empty:
        print("No data fetched. Please check the ticker symbol or your internet connection.")
    else:
        print("Data fetched successfully!")
        print(f"Data range: {data.index.min()} to {data.index.max()}")

        required = {'Open', 'High', 'Low', 'Close'}
        if not required.issubset(set(data.columns)):
            print("Detected columns:", list(data.columns))
            raise ValueError("`data` must be a pandas.DataFrame with columns 'Open', 'High', 'Low', 'Close' (and optionally 'Volume').")

        bt = Backtest(data, SmaCross,
                      cash=100000,  # Starting with ₹1,00,000
                      commission=.002) # 0.2% commission per trade

        print("\nRunning backtest...")
        stats = bt.run()
        print("\nBacktest complete!")

        print("\n--- Backtest Results ---")
        print(stats)

        print("\nGenerating plot... (A new browser window will open)")
        bt.plot()