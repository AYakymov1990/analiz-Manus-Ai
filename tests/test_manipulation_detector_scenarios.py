import pandas as pd
import numpy as np

from src.analyzers.manipulation_detector import ManipulationDetector
from src.core.data_structures import ManipulationType


def _ohlc_from_close(close: pd.Series) -> pd.DataFrame:
    o = close.shift(1).fillna(close.iloc[0])
    h = np.maximum(o, close) + 0.2
    l = np.minimum(o, close) - 0.2
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close}, index=close.index)


def test_stop_hunt_above_bsl_zone():
    idx = pd.date_range("2024-01-01", periods=60, freq="15min")
    base = pd.Series(3863.0, index=idx)
    close = base.copy()
    # небольшие колебания
    close += np.sin(np.linspace(0, 6 * np.pi, len(idx))) * 1.0
    # финальный рывок выше BSL и быстрый возврат
    close.iloc[-4] = 3920.0
    close.iloc[-3] = 3930.0  # breakout above
    close.iloc[-2] = 3922.0  # return
    close.iloc[-1] = 3915.0

    df = _ohlc_from_close(close)
    context = {
        "current_price": float(df["Close"].iloc[-1]),
        "liquidity_zones": {"BSL": [3926.62], "SSL": []},
        "fibonacci_zone": "extension_above",
    }

    det = ManipulationDetector()
    res = det.detect_manipulation(df, context)

    assert res.manipulation_type in (ManipulationType.STOP_HUNT_ABOVE, ManipulationType.LIQUIDITY_GRAB)
    assert res.details["structure"]["broke_above"] is True


def test_false_breakout_without_volume():
    idx = pd.date_range("2024-01-01", periods=80, freq="15min")
    close = pd.Series(3810.0, index=idx)
    close += np.sin(np.linspace(0, 4 * np.pi, len(idx))) * 0.5
    # пробой сопротивления чуть выше и тут же откат без импульса
    close.iloc[-3] = 3815.0
    close.iloc[-2] = 3816.0  # breakout small
    close.iloc[-1] = 3813.5  # back below

    df = _ohlc_from_close(close)
    # без volume, чтобы не было спайков по zscore
    context = {
        "current_price": float(df["Close"].iloc[-1]),
        "liquidity_zones": {"BSL": [], "SSL": []},
        "fibonacci_zone": "premium",
    }

    det = ManipulationDetector()
    res = det.detect_manipulation(df, context)

    assert res.manipulation_type in (ManipulationType.FALSE_BREAKOUT, ManipulationType.NO_MANIPULATION)

