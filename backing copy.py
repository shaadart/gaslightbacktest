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
        # - If short SMA crosses above long SMA -> open a long position (if none)
        # - If long SMA crosses below long SMA -> close existing long position
        if crossover(self.sma1, self.sma2):
            # Only open a new long if we don't already have a position
            if not self.position:
                # Use a small fixed size to avoid insufficient margin issues
                self.buy(size=1)
        elif crossover(self.sma2, self.sma1):
            # If we have a position, close it on cross down
            if self.position:
                self.position.close()


# =============================================================================
# 3. PREPARE THE DATA
# =============================================================================
print("Fetching historical stock data for Reliance Industries...")
# Fetch data for the last 5 years
data = yf.download('RELIANCE.NS', start='2020-01-01', end=None)

# -----------------------------------------------------------------------------
# THE FIX: Flatten the column headers if they are a MultiIndex
# Recent yfinance versions can return multi-level columns. We simplify them.
# -----------------------------------------------------------------------------
if isinstance(data.columns, pd.MultiIndex):
    # Choose the level that contains OHLC-like names (Open/High/Low/Close/Adj Close/Volume).
    # Different yfinance versions/orderings put ticker/attribute in different levels.
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
        # Fallback: keep the last level
        try:
            data.columns = data.columns.get_level_values(-1)
        except Exception:
            # Final fallback: join tuple parts into a single string
            data.columns = [" ".join([str(p) for p in c if p is not None]) if isinstance(c, tuple) else str(c) for c in data.columns]

# Ensure column names are plain strings
data.columns = [c if isinstance(c, str) else str(c) for c in data.columns]

# If 'Close' is missing but 'Adj Close' exists, use adjusted close as Close
if 'Close' not in data.columns and 'Adj Close' in data.columns:
    data['Close'] = data['Adj Close']

# Some providers return lowercase column names - normalize common names
col_map = {c: c for c in data.columns}
lower_to_col = {c.lower(): c for c in data.columns}
for needed in ('open', 'high', 'low', 'close', 'volume'):
    if needed in lower_to_col and lower_to_col[needed] != needed.title():
        col_map[lower_to_col[needed]] = needed.title()
if col_map and col_map != {c: c for c in data.columns}:
    data = data.rename(columns=col_map)
# -----------------------------------------------------------------------------

if data.empty:
    print("No data fetched. Please check the ticker symbol or your internet connection.")
else:
    print("Data fetched successfully!")
    print(f"Data range: {data.index.min()} to {data.index.max()}")

    # =============================================================================
    # 4. CONFIGURE AND RUN THE BACKTEST (The "Engine")
    # =============================================================================
    # Validate that required columns exist for Backtest
    required = {'Open', 'High', 'Low', 'Close'}
    if not required.issubset(set(data.columns)):
        print("Detected columns:", list(data.columns))
        raise ValueError("`data` must be a pandas.DataFrame with columns 'Open', 'High', 'Low', 'Close' (and optionally 'Volume').\nPlease check the downloaded data and column names.")

    bt = Backtest(data, SmaCross,
                  cash=100000,  # Starting with ₹1,00,000
                  commission=.002) # 0.2% commission per trade

    print("\nRunning backtest...")
    stats = bt.run()
    print("\nBacktest complete!")

    # =============================================================================
    # 5. REVIEW THE RESULTS
    # =============================================================================
    print("\n--- Backtest Results ---")
    print(stats)

    print("\nGenerating plot... (A new browser window will open)")
    bt.plot()