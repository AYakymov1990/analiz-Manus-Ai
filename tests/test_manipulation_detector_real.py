import os
import pytest
import pandas as pd

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

from src.analyzers.manipulation_detector import ManipulationDetector, ManipulationResult


def _fetch(symbol: str = "GC=F", period: str = "3d", interval: str = "15m") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    return df


def _ensure_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[symbol]
        except Exception:
            df = df.droplevel(-1, axis=1)
    return df[["Open", "High", "Low", "Close"]]


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Интеграционный тест требует сети; включить с RUN_INTEGRATION_TESTS=1",
)
def test_manipulation_detector_real_data():
    m15 = _fetch("GC=F", "3d", "15m")
    assert not m15.empty
    m15 = _ensure_columns(m15, "GC=F")
    m15.index = pd.to_datetime(m15.index)

    setup_context = {}

    detector = ManipulationDetector()
    res: ManipulationResult = detector.detect_manipulation(m15, setup_context)

    assert res.manipulation_type is not None
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.details, dict)


