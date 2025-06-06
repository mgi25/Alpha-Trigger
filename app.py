import os
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:  # library not available on all systems
    mt5 = None

TIMEFRAME_M5 = mt5.TIMEFRAME_M5 if mt5 else 5
import ccxt
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


PASSWORD = "Mgi@2005"
SERVER = "Exness-MT5Trial7"
MT5_SYMBOL = "XAUUSDm"
BINANCE_SYMBOL = "XAU/USDT"


def mt5_login(account: int) -> bool:
    """Initialize connection to MetaTrader 5."""
    if mt5 is None:
        raise ImportError("MetaTrader5 package is not installed")
    if not mt5.initialize(server=SERVER, login=account, password=PASSWORD):
        raise RuntimeError(f"initialize() failed: {mt5.last_error()}")
    return True


def fetch_ohlcv_mt5(
    account: int, symbol: str = MT5_SYMBOL, timeframe=TIMEFRAME_M5, limit: int = 500
):
    """Fetch OHLCV data from Exness via MetaTrader 5."""
    mt5_login(account)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)
    mt5.shutdown()
    if rates is None:
        raise RuntimeError("No rates returned from MT5")
    df = pd.DataFrame(rates)
    df["timestamp"] = df["time"] * 1000
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv(exchange, symbol=BINANCE_SYMBOL, timeframe="5m", limit=500):
    """Fetch OHLCV data from Binance using ccxt (fallback)."""
    exchange.load_markets()
    if symbol not in exchange.symbols:
        raise ValueError(f"Symbol {symbol} not available on {exchange.id}")
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(
        data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )


def add_indicators(df):
    """Add VWAP, ATR, RSI, MACD histogram and other features."""
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

    atr_indicator = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr_indicator.average_true_range()
    df["atr_change"] = df["atr"].pct_change()
    df["atr_ma"] = df["atr"].rolling(window=10).mean()
    df["atr_vs_avg"] = df["atr"] / df["atr_ma"]

    rsi_indicator = RSIIndicator(df["close"], window=14)
    df["rsi"] = rsi_indicator.rsi()

    macd_indicator = MACD(df["close"])
    df["macd_hist"] = macd_indicator.macd_diff()

    df["vwap_distance"] = df["close"] - df["vwap"]
    df["vwap_dist_pct"] = df["vwap_distance"] / df["close"]
    df["time_bucket"] = pd.to_datetime(df["timestamp"], unit="ms").dt.hour // 4
    return df.dropna()


def label_breakouts(df, tp=0.002, sl=0.001, lookahead: int = 3):
    """Vectorized breakout labeling using lookahead candles."""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - lookahead):
        entry = closes[i]
        target_up = entry * (1 + tp)
        target_down = entry * (1 - sl)
        future_highs = highs[i + 1 : i + 1 + lookahead]
        future_lows = lows[i + 1 : i + 1 + lookahead]

        up_hit_idx = (
            np.argmax(future_highs >= target_up)
            if np.any(future_highs >= target_up)
            else lookahead
        )
        down_hit_idx = (
            np.argmax(future_lows <= target_down)
            if np.any(future_lows <= target_down)
            else lookahead
        )

        labels[i] = 1 if up_hit_idx < down_hit_idx else 0

    df["label"] = labels
    return df.dropna()


def prepare_features(df):
    features = df[
        [
            "close",
            "volume",
            "atr",
            "atr_change",
            "atr_vs_avg",
            "vwap_dist_pct",
            "rsi",
            "macd_hist",
            "time_bucket",
        ]
    ]
    return features, df["label"]


def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Validation accuracy: {accuracy:.2f}")
    joblib.dump(model, "alpha_model.joblib")
    return model


def load_model(path="alpha_model.joblib"):
    return joblib.load(path)


def predict_breakout(model, row):
    features = row[
        [
            "close",
            "volume",
            "atr",
            "atr_change",
            "atr_vs_avg",
            "vwap_dist_pct",
            "rsi",
            "macd_hist",
            "time_bucket",
        ]
    ].values.reshape(1, -1)
    prob = model.predict_proba(features)[0, 1]
    return prob


def main():
    account = int(os.environ.get("MT5_LOGIN", "0"))
    try:
        df = fetch_ohlcv_mt5(account)
    except Exception as ex:
        print(f"MT5 fetch failed: {ex}. Falling back to Binance via ccxt.")
        exchange = ccxt.binance()
        df = fetch_ohlcv(exchange)
    df = add_indicators(df)
    df = label_breakouts(df)
    model = train_model(df)
    last_row = df.iloc[-1]
    prob = predict_breakout(model, last_row)
    print(f"Probability of profitable breakout: {prob:.2%}")


if __name__ == "__main__":
    main()
