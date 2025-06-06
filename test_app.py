import pandas as pd
from app import add_indicators


def test_add_indicators():
    n = 20
    df = pd.DataFrame(
        {
            "timestamp": range(n),
            "open": range(1, n + 1),
            "high": range(2, n + 2),
            "low": [x - 0.5 for x in range(1, n + 1)],
            "close": [x + 0.5 for x in range(1, n + 1)],
            "volume": [100] * n,
        }
    )
    out = add_indicators(df.copy())
    assert "vwap" in out.columns
    assert "atr" in out.columns
    assert not out[["vwap", "atr"]].isna().any().any()
