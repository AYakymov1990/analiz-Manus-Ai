import os
import json
import time
from typing import Dict

import pandas as pd
import pytest

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.analyzers.retail_analyzer import RetailBehaviorAnalyzer


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


def _fetch(symbol: str = "GC=F", period: str = "59d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="15m", auto_adjust=False, progress=False, threads=False)
    if df.empty:
        for iv in ["30m", "60m"]:
            df = yf.download(symbol, period="3mo", interval=iv, auto_adjust=False, progress=False, threads=False)
            if not df.empty:
                break
    return df


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Требуется сеть; включить с RUN_INTEGRATION_TESTS=1",
)
def test_real_data_end_to_end(tmp_path):
    start = time.time()
    symbol = "GC=F"
    raw = _fetch(symbol, "59d")
    assert not raw.empty, "No real data"
    raw = _ensure_columns(raw, symbol)
    raw.index = pd.to_datetime(raw.index)

    data_m15 = raw.copy()
    data_h4 = _resample(raw, "4h")
    data_1d = _resample(raw, "1D")

    # Окна для анализа (исключаем старые экстремумы)
    data_1d_win = data_1d.loc[data_1d.index >= data_1d.index[-1] - pd.Timedelta(days=35)]
    data_h4_win = data_h4.loc[data_h4.index >= data_h4.index[-1] - pd.Timedelta(days=14)]

    current_price = float(data_m15["Close"].iloc[-1])

    # Stage 1: Structure
    sa = MultiTimeframeStructureAnalyzer(symbol)
    structures = sa.analyze_all_timeframes(data_1d_win, data_h4_win, data_m15)
    assert structures["1D"].last_swing_high.price >= structures["1D"].last_swing_low.price

    # Swing уровни должны реально присутствовать в 1D окне
    def _exists(val_series: pd.Series, p: float, tol: float = 1e-6) -> bool:
        return (val_series.sub(p).abs() <= tol).any()

    assert _exists(data_1d_win["High"], structures["1D"].last_swing_high.price), "1D swing high not in OHLC window"
    assert _exists(data_1d_win["Low"], structures["1D"].last_swing_low.price), "1D swing low not in OHLC window"

    # Stage 2: Fibonacci
    fa = MultiTimeframeFibonacciAnalyzer()
    fibs = fa.analyze_all_timeframes(structures, current_price)
    assert fibs["1D"].swing_high >= fibs["1D"].swing_low
    if fibs["1D"].current_zone.name.startswith("EXTENSION"):
        assert fibs["1D"].retracement_level > 1.0

    # Stage 3: Retail
    ra = RetailBehaviorAnalyzer(symbol)
    retail = ra.analyze_retail_behavior(data_1d_win, fibs, structures)
    assert len(retail["support_resistance_levels"]) > 0, "S/R must not be empty on real data"
    assert len(retail["liquidity_zones"]) > 0

    # Сохранить расширенный JSON
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "real_chain"))
    os.makedirs(out_dir, exist_ok=True)

    sr_1d = [
        {"price": lvl.price, "touches": lvl.touches, "strength": lvl.strength, "type": lvl.level_type}
        for lvl in retail["support_resistance_levels"]
    ]
    # 4H уровни: используем метод с инъекцией якорей
    levels_4h = ra.find_support_resistance_levels_h4(data_h4_win, structures)
    sr_4h = [
        {"price": lvl.price, "touches": lvl.touches, "strength": lvl.strength, "type": lvl.level_type}
        for lvl in levels_4h
    ]

    report: Dict = {
        "symbol": symbol,
        "data_windows": {
            "1D": [str(data_1d_win.index[0]), str(data_1d_win.index[-1])],
            "4H": [str(data_h4_win.index[0]), str(data_h4_win.index[-1])],
        },
        "current_price": current_price,
        "fibonacci": {
            tf: {
                "zone": v.current_zone.value,
                "retracement": v.retracement_level,
                "swing_high": v.swing_high,
                "swing_low": v.swing_low,
            }
            for tf, v in fibs.items()
        },
        "support_resistance": {"1D": sr_1d, "4H": sr_4h},
        "liquidity_zones": [
            {
                "price": z.price,
                "type": z.zone_type,
                "strength": z.strength,
                "volume": z.estimated_volume,
                "logic": z.retail_logic,
            }
            for z in retail["liquidity_zones"]
        ],
        "elapsed_sec": round(time.time() - start, 2),
    }

    with open(os.path.join(out_dir, "real_chain_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


