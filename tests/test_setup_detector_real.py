import os
import pytest
import pandas as pd

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.analyzers.retail_analyzer import RetailBehaviorAnalyzer
from src.analyzers.setup_detector import SetupDetector, SetupResult


def _ensure_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[symbol]
        except Exception:
            df = df.droplevel(-1, axis=1)
    return df[["Open", "High", "Low", "Close"]]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    res = pd.concat([o, h, l, c], axis=1).dropna()
    res.columns = ["Open", "High", "Low", "Close"]
    return res


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Требуется сеть; включить с RUN_INTEGRATION_TESTS=1",
)
def test_setup_detector_on_real_chain():
    symbol = "GC=F"
    raw = yf.download(symbol, period="59d", interval="15m", auto_adjust=False, progress=False, threads=False)
    assert not raw.empty
    raw = _ensure_columns(raw, symbol)
    raw.index = pd.to_datetime(raw.index)

    data_m15 = raw.copy()
    data_1d = _resample(raw, "1D")
    data_h4 = _resample(raw, "4h")

    # Окна анализа
    data_1d_win = data_1d.loc[data_1d.index >= data_1d.index[-1] - pd.Timedelta(days=35)]
    data_h4_win = data_h4.loc[data_h4.index >= data_h4.index[-1] - pd.Timedelta(days=14)]
    current_price = float(data_m15["Close"].iloc[-1])

    sa = MultiTimeframeStructureAnalyzer(symbol)
    structures = sa.analyze_all_timeframes(data_1d_win, data_h4_win, data_m15)
    fa = MultiTimeframeFibonacciAnalyzer()
    fibs = fa.analyze_all_timeframes(structures, current_price)
    ra = RetailBehaviorAnalyzer(symbol)
    retail = ra.analyze_retail_behavior(data_1d_win, fibs, structures)

    detector = SetupDetector()
    setup: SetupResult = detector.detect_setup(structures, fibs, retail)

    assert setup.setup_type is not None
    assert 0.0 <= setup.confidence <= 1.0
    assert isinstance(setup.conditions, dict)


