import os
import pytest
import numpy as np
import pandas as pd

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

import matplotlib

matplotlib.use("Agg")  # Без GUI-бэкенда
import matplotlib.pyplot as plt
pytest.importorskip("mplfinance")
import mplfinance as mpf  # type: ignore

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer


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
def test_xauusd_integration_and_visualization(tmp_path):
    # Загружаем минутные данные XAUUSD (Gold futures proxy GC=F) и строим 1D/4H/15M
    ticker = "GC=F"
    data_raw = yf.download(
        ticker,
        period="59d",  # 15m доступно только до 60 дней
        interval="15m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data_raw.empty:
        # Фолбэк на более крупный интервал
        for iv in ["30m", "60m"]:
            data_raw = yf.download(
                ticker,
                period="3mo",
                interval=iv,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not data_raw.empty:
                break
    assert not data_raw.empty, "Нет данных от yfinance"

    # Обработка мультииндекса колонок (yfinance для одиночного тикера иногда отдает MultiIndex)
    if isinstance(data_raw.columns, pd.MultiIndex):
        try:
            data_raw = data_raw[ticker]
        except Exception:
            data_raw = data_raw.droplevel(-1, axis=1)

    data_raw = data_raw[["Open", "High", "Low", "Close"]]
    data_raw.index = pd.to_datetime(data_raw.index)

    data_m15 = data_raw.copy()
    data_h4 = _resample(data_raw, "4h")
    data_1d = _resample(data_raw, "1D")

    analyzer = MultiTimeframeStructureAnalyzer("XAUUSD")
    res = analyzer.analyze_all_timeframes(data_1d, data_h4, data_m15)

    assert set(res.keys()) == {"1D", "4H", "15M"}
    # Визуализация свингов на 1D свечами
    swings_1d = analyzer.detect_swing_points(data_1d, "1D")
    swing_high_series = pd.Series(index=data_1d.index, dtype=float)
    swing_low_series = pd.Series(index=data_1d.index, dtype=float)
    for s in swings_1d:
        if s.type == "high" and s.timestamp in swing_high_series.index:
            swing_high_series.loc[s.timestamp] = s.price
        elif s.type == "low" and s.timestamp in swing_low_series.index:
            swing_low_series.loc[s.timestamp] = s.price

    ap = [
        mpf.make_addplot(swing_high_series, type="scatter", markersize=30, marker="^", color="red"),
        mpf.make_addplot(swing_low_series, type="scatter", markersize=30, marker="v", color="green"),
    ]

    out_path = tmp_path / "xauusd_swings_1d_candles.png"
    mpf.plot(
        data_1d,
        type="candle",
        style="yahoo",
        addplot=ap,
        figscale=1.2,
        volume=False,
        savefig=str(out_path),
    )
    # Дополнительно сохраняем в output/ проекта для постоянного доступа
    proj_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    try:
        os.makedirs(proj_out_dir, exist_ok=True)
        mpf.plot(
            data_1d,
            type="candle",
            style="yahoo",
            addplot=ap,
            figscale=1.2,
            volume=False,
            savefig=os.path.join(proj_out_dir, "xauusd_swings_1d_candles.png"),
        )
    except Exception:
        pass
    assert out_path.exists(), "Картинка не сохранена"


