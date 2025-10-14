from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
from backtesting import Backtest
import traceback

# Import strategies from backing.py (we'll import the module and reference classes)
import backing as backing_module

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class BacktestRequest(BaseModel):
    symbol: str
    start: str | None = None
    end: str | None = None
    strategies: dict  # e.g., {"SMA": {"n1":10, "n2":30}, "RSI": {"period":14}}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Offer a list of available strategies from backing_module
    available = [c for c in dir(backing_module) if c.endswith('Strategy')]
    return templates.TemplateResponse('index.html', {"request": request, "strategies": available})

@app.post("/run-backtest")
async def run_backtest(req: BacktestRequest):
    try:
        symbol = req.symbol
        start = req.start
        end = req.end
        # Fetch data
        data = yf.download(symbol, start=start, end=end)
        # Reuse the same normalization logic in backing.py
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
        # Normalize casing
        col_map = {c: c for c in data.columns}
        lower_to_col = {c.lower(): c for c in data.columns}
        for needed in ('open', 'high', 'low', 'close', 'volume'):
            if needed in lower_to_col and lower_to_col[needed] != needed.title():
                col_map[lower_to_col[needed]] = needed.title()
        if col_map and col_map != {c: c for c in data.columns}:
            data = data.rename(columns=col_map)

        # Validate columns
        required = {'Open', 'High', 'Low', 'Close'}
        if not required.issubset(set(data.columns)):
            return JSONResponse({"error": "Data missing required OHLC columns.", "detected": list(data.columns)}, status_code=400)

        results = {}
        for name, params in req.strategies.items():
            # Map name to Strategy class in backing_module
            strat_cls = getattr(backing_module, name, None)
            if strat_cls is None:
                results[name] = {"error": "Strategy not found"}
                continue
            # Apply params to strategy class if provided
            for k, v in params.items():
                if hasattr(strat_cls, k):
                    setattr(strat_cls, k, v)
            bt = Backtest(data, strat_cls, cash=100000, commission=.002)
            stats = bt.run()
            # pick a subset of stats to return
            results[name] = {
                'Return [%]': float(stats['Return [%]']),
                'Win Rate [%]': float(stats['Win Rate [%]']) if not pd.isna(stats['Win Rate [%]']) else None,
                'Sharpe Ratio': float(stats['Sharpe Ratio']) if not pd.isna(stats['Sharpe Ratio']) else None,
                'Max. Drawdown [%]': float(stats['Max. Drawdown [%]']) if not pd.isna(stats['Max. Drawdown [%]']) else None,
                'Equity Final': float(stats['Equity Final [$]']) if not pd.isna(stats['Equity Final [$]']) else None,
                'Trades': int(stats['# Trades']) if not pd.isna(stats['# Trades']) else 0,
            }
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)
