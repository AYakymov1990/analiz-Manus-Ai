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
from src.analyzers.manipulation_detector import ManipulationDetector
from src.core.context_builder import AnalysisResults, build_manus_context


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
def test_build_manus_context_and_save_json(tmp_path):
    symbol = "GC=F"
    raw = yf.download(symbol, period="59d", interval="15m", auto_adjust=False, progress=False, threads=False)
    assert not raw.empty, "Нет данных от yfinance"
    raw = _ensure_columns(raw, symbol)
    raw.index = pd.to_datetime(raw.index)

    data_m15 = raw.copy()
    data_h4 = _resample(raw, "4h")
    data_1d = _resample(raw, "1D")

    # Узкие окна для консистентности с real chain
    win_1d = data_1d.loc[data_1d.index >= data_1d.index[-1] - pd.Timedelta(days=35)]
    win_4h = data_h4.loc[data_h4.index >= data_h4.index[-1] - pd.Timedelta(days=14)]

    current_price = float(data_m15["Close"].iloc[-1])

    sa = MultiTimeframeStructureAnalyzer(symbol)
    structures = sa.analyze_all_timeframes(win_1d, win_4h, data_m15)

    fa = MultiTimeframeFibonacciAnalyzer()
    fibs = fa.analyze_all_timeframes(structures, current_price)

    ra = RetailBehaviorAnalyzer(symbol)
    retail = ra.analyze_retail_behavior(win_1d, fibs, structures)

    sd = SetupDetector()
    setup = sd.detect_setup(structures, fibs, retail)

    md = ManipulationDetector()
    manip = md.detect_manipulation(data_m15.tail(500), {
        "current_price": current_price,
        "liquidity_zones": {
            "BSL": [z.price for z in retail["liquidity_zones"] if z.zone_type == "BSL"],
            "SSL": [z.price for z in retail["liquidity_zones"] if z.zone_type == "SSL"],
        },
        "fibonacci_zone": fibs["1D"].current_zone.value,
    })

    analysis = AnalysisResults(
        symbol=symbol,
        current_price=current_price,
        data_windows={
            "1D": (str(win_1d.index[0]), str(win_1d.index[-1])),
            "4H": (str(win_4h.index[0]), str(win_4h.index[-1])),
        },
        structures=structures,
        fibonacci=fibs,
        retail=retail,
        setup=setup,
        manipulation=manip,
    )

    ctx = build_manus_context(symbol, analysis)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "real_chain"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "manus_ai_context.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    assert os.path.exists(out_path)
    assert ctx["metadata"]["symbol"] == symbol
    assert "market_structure" in ctx and "fibonacci_analysis" in ctx and "retail_behavior" in ctx

