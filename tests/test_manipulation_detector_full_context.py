import os
import json
import pytest
import pandas as pd

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.analyzers.retail_analyzer import RetailBehaviorAnalyzer
from src.analyzers.setup_detector import SetupDetector
from src.analyzers.manipulation_detector import ManipulationDetector, ManipulationResult


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
    reason="Интеграционный тест требует сети; включить с RUN_INTEGRATION_TESTS=1",
)
def test_manipulation_detector_with_full_context(tmp_path):
    symbol = "GC=F"
    raw = yf.download(symbol, period="5d", interval="15m", auto_adjust=False, progress=False, threads=False)
    assert not raw.empty, "Нет данных от yfinance"
    raw = _ensure_columns(raw, symbol)
    raw.index = pd.to_datetime(raw.index)

    data_m15 = raw.copy()
    data_h4 = _resample(raw, "4h")
    data_1d = _resample(raw, "1D")

    current_price = float(data_m15["Close"].iloc[-1])

    # Stage 1-2: Structure
    sa = MultiTimeframeStructureAnalyzer(symbol)
    structures = sa.analyze_all_timeframes(data_1d, data_h4, data_m15)

    # Stage 3: Fibonacci
    fa = MultiTimeframeFibonacciAnalyzer()
    fibs = fa.analyze_all_timeframes(structures, current_price)

    # Stage 4: Retail
    ra = RetailBehaviorAnalyzer(symbol)
    retail = ra.analyze_retail_behavior(data_1d, fibs, structures)

    # Stage 5: Setup Detector
    sd = SetupDetector()
    setup = sd.detect_setup(structures, fibs, retail)

    # Build full context for Manipulation Detector
    bsl = [z.price for z in retail["liquidity_zones"] if z.zone_type == "BSL"]
    ssl = [z.price for z in retail["liquidity_zones"] if z.zone_type == "SSL"]

    full_context = {
        "current_price": current_price,
        "swing_levels": {
            "1D_high": float(structures["1D"].last_swing_high.price),
            "1D_low": float(structures["1D"].last_swing_low.price),
            "4H_high": float(structures["4H"].last_swing_high.price),
            "4H_low": float(structures["4H"].last_swing_low.price),
        },
        "liquidity_zones": {"BSL": bsl, "SSL": ssl},
        "fibonacci_zone": fibs["1D"].current_zone.value,
        "setup_type": setup.setup_type.value,
        "market_structure": structures["1D"].direction.value,
    }

    # Stage 6: Manipulation
    md = ManipulationDetector()
    res: ManipulationResult = md.detect_manipulation(data_m15.tail(500), full_context)

    # Save JSON for review
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "real_chain"))
    os.makedirs(out_dir, exist_ok=True)
    output = {
        "manipulation_type": res.manipulation_type.value,
        "confidence": res.confidence,
        "details": res.details,
        "context": full_context,
    }
    with open(os.path.join(out_dir, "manipulation_full_context.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Basic assertions
    assert res.manipulation_type is not None
    assert 0.0 <= res.confidence <= 1.0
