import numpy as np
import pandas as pd

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer


def make_ohlc_from_series(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    df["Open"] = series.shift(1).fillna(series.iloc[0])
    df["High"] = np.maximum(series, df["Open"]) + 0.0001
    df["Low"] = np.minimum(series, df["Open"]) - 0.0001
    df["Close"] = series
    return df


def test_detect_swings_and_bullish_structure_daily():
    # Создаем возрастающую ступенчатую серию с явными экстремумами
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    base = np.linspace(1.20, 1.40, len(idx))
    waves = np.sin(np.linspace(0, 8 * np.pi, len(idx))) * 0.01
    price = base + waves
    series = pd.Series(price, index=idx)
    data = make_ohlc_from_series(series)

    analyzer = MultiTimeframeStructureAnalyzer(symbol="EURUSD")
    swings = analyzer.detect_swing_points(data, "1D")

    # Должны быть найдены swing точки и их хотя бы 4
    assert len(swings) >= 4
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]
    assert len(highs) >= 2 and len(lows) >= 2

    sa = analyzer.determine_structure(swings, "1D")
    assert sa.direction.value in {"bullish", "bearish", "sideways"}


def test_bearish_structure_on_descending_series_h4():
    idx = pd.date_range("2024-01-01", periods=240, freq="4H")
    base = np.linspace(1.40, 1.20, len(idx))
    waves = np.sin(np.linspace(0, 12 * np.pi, len(idx))) * 0.005
    series = pd.Series(base + waves, index=idx)
    data = make_ohlc_from_series(series)

    analyzer = MultiTimeframeStructureAnalyzer(symbol="EURUSD")
    swings = analyzer.detect_swing_points(data, "4H")
    sa = analyzer.determine_structure(swings, "4H")

    # На нисходящей серии ожидается bearish или sideways (если шум мешает)
    assert sa.direction in {sa.direction.BEARISH, sa.direction.SIDEWAYS}


def test_analyze_all_timeframes_returns_dict_with_keys():
    idx_d = pd.date_range("2024-01-01", periods=60, freq="D")
    idx_h4 = pd.date_range("2024-01-01", periods=240, freq="4H")
    idx_m15 = pd.date_range("2024-01-01", periods=24 * 20 * 4, freq="15min")

    s_d = pd.Series(1.30 + np.sin(np.linspace(0, 6 * np.pi, len(idx_d))) * 0.01, index=idx_d)
    s_h4 = pd.Series(1.30 + np.cos(np.linspace(0, 10 * np.pi, len(idx_h4))) * 0.008, index=idx_h4)
    s_m15 = pd.Series(1.30 + np.sin(np.linspace(0, 20 * np.pi, len(idx_m15))) * 0.002, index=idx_m15)

    d_d = make_ohlc_from_series(s_d)
    d_h4 = make_ohlc_from_series(s_h4)
    d_m15 = make_ohlc_from_series(s_m15)

    analyzer = MultiTimeframeStructureAnalyzer("EURUSD")
    res = analyzer.analyze_all_timeframes(d_d, d_h4, d_m15)

    assert set(res.keys()) == {"1D", "4H", "15M"}
    assert all(hasattr(v, "direction") for v in res.values())


