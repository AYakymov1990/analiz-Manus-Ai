import pandas as pd

from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.core.data_structures import (
    StructureAnalysis,
    StructureDirection,
    SwingPoint,
    FibonacciZone,
)
from datetime import datetime


def make_structure(high: float, low: float, timeframe: str = "1D") -> StructureAnalysis:
    ts = pd.Timestamp(datetime(2024, 1, 1))
    sp_high = SwingPoint(timestamp=ts, price=high, type="high", timeframe=timeframe, strength=0.7)
    sp_low = SwingPoint(timestamp=ts, price=low, type="low", timeframe=timeframe, strength=0.6)
    return StructureAnalysis(
        timeframe=timeframe,
        direction=StructureDirection.BULLISH,
        last_swing_high=sp_high,
        last_swing_low=sp_low,
        structure_strength=0.75,
        break_level=low,
        confidence=0.8,
    )


def test_calculate_fibonacci_retracement_levels_and_zone():
    analyzer = MultiTimeframeFibonacciAnalyzer()
    swing_high, swing_low = 1.40, 1.20
    current_price = 1.31
    res = analyzer.calculate_fibonacci_retracement(swing_high, swing_low, current_price, "1D")

    assert res.timeframe == "1D"
    assert abs(res.key_levels["0.5"] - 1.30) < 1e-9
    assert res.current_zone in {
        FibonacciZone.PREMIUM,
        FibonacciZone.DISCOUNT,
        FibonacciZone.OTE,
        FibonacciZone.EQUILIBRIUM,
        FibonacciZone.EXTENSION_ABOVE,
        FibonacciZone.EXTENSION_BELOW,
    }
    assert 0.0 <= res.retracement_level <= 1.0


def test_determine_ote_zone():
    analyzer = MultiTimeframeFibonacciAnalyzer()
    # Цена в диапазоне между 0.618 и 0.786
    res = analyzer.calculate_fibonacci_retracement(1.40, 1.20, 1.34, "4H")
    assert res.current_zone is FibonacciZone.OTE


def test_extension_above_and_below():
    analyzer = MultiTimeframeFibonacciAnalyzer()
    # bullish leg, цена выше high => extension above
    res1 = analyzer.calculate_fibonacci_retracement(1.40, 1.20, 1.45, "1D")
    assert res1.current_zone is FibonacciZone.EXTENSION_ABOVE
    assert res1.retracement_level > 1.0

    # bearish leg, цена ниже low => extension below
    res2 = analyzer.calculate_fibonacci_retracement(1.20, 1.40, 1.15, "1D")
    assert res2.current_zone is FibonacciZone.EXTENSION_BELOW
    assert res2.retracement_level > 1.0 or res2.retracement_level < 0.0


def test_analyze_all_timeframes_from_structures():
    analyzer = MultiTimeframeFibonacciAnalyzer()
    structures = {
        "1D": make_structure(1.40, 1.20, "1D"),
        "4H": make_structure(1.38, 1.22, "4H"),
        "15M": make_structure(1.35, 1.27, "15M"),
    }
    current_price = 1.315
    res = analyzer.analyze_all_timeframes(structures, current_price)

    assert set(res.keys()) == {"1D", "4H", "15M"}
    assert all(0.0 <= v.retracement_level <= 1.0 for v in res.values())


def test_detect_ote_consolidation_basic():
    analyzer = MultiTimeframeFibonacciAnalyzer()
    # Синтетические данные, цена колеблется внутри [1.30, 1.32]
    idx = pd.date_range("2024-01-01", periods=30, freq="H")
    close = pd.Series(1.30, index=idx).astype(float)
    close.iloc[::3] = 1.315
    df = pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 0.002,
        "Low": close - 0.002,
        "Close": close,
    }, index=idx)

    assert analyzer.detect_ote_consolidation(df, (1.30, 1.32)) is True


