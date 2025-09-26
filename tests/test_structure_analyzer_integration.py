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
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer


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

    # Fibonacci overlay based on 1D structure
    fib_an = MultiTimeframeFibonacciAnalyzer()
    current_price = float(data_m15["Close"].iloc[-1]) if len(data_m15) else float(data_1d["Close"].iloc[-1])
    fib1d = fib_an.calculate_fibonacci_retracement(
        res["1D"].last_swing_high.price,
        res["1D"].last_swing_low.price,
        current_price,
        "1D",
    )

    out_path = tmp_path / "xauusd_swings_1d_candles.png"
    fig, axlist = mpf.plot(
        data_1d,
        type="candle",
        style="yahoo",
        addplot=ap,
        figscale=1.2,
        volume=False,
        returnfig=True,
    )
    ax = axlist[0]

    # Determine pivot (last swing timestamp) to draw levels to the right only
    ts_high = swing_high_series.dropna().index.max() if swing_high_series.notna().any() else None
    ts_low = swing_low_series.dropna().index.max() if swing_low_series.notna().any() else None
    pivot_ts = None
    if ts_high is not None and ts_low is not None:
        pivot_ts = max(ts_high, ts_low)
    elif ts_high is not None:
        pivot_ts = ts_high
    elif ts_low is not None:
        pivot_ts = ts_low
    start_idx = 0
    if pivot_ts is not None:
        try:
            start_idx = int(np.searchsorted(data_1d.index.values, pivot_ts))
            start_idx = max(0, min(start_idx, len(data_1d) - 1))
        except Exception:
            start_idx = int(len(data_1d) * 0.5)
    x_right = data_1d.index[start_idx:]

    # Plot fib lines from pivot to right, with labels
    for k, y in fib1d.key_levels.items():
        if len(x_right) >= 2:
            ax.plot(x_right, [y] * len(x_right), color="#666", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(
            x_right[0] if len(x_right) else data_1d.index[0],
            y,
            k,
            fontsize=8,
            color="#222",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )

    # Zone fills to the right of pivot only, and labels
    y_0_5 = fib1d.key_levels.get("0.5")
    y_0618 = fib1d.key_levels.get("0.618")
    y_0786 = fib1d.key_levels.get("0.786")
    if len(x_right) >= 2:
        if y_0_5 is not None:
            ax.fill_between(x_right, y_0_5, ax.get_ylim()[1], color="#f8c3c3", alpha=0.12)
            ax.fill_between(x_right, ax.get_ylim()[0], y_0_5, color="#c3f3c3", alpha=0.18)
        if y_0618 is not None and y_0786 is not None:
            ax.fill_between(x_right, y_0618, y_0786, color="#fff3b0", alpha=0.25)
        mid_x = x_right[len(x_right) // 2]
        if y_0_5 is not None:
            ax.text(mid_x, (y_0_5 + ax.get_ylim()[1]) / 2, "Premium", fontsize=9, color="#aa0000", ha="center", va="center", alpha=0.8)
            ax.text(mid_x, (ax.get_ylim()[0] + y_0_5) / 2, "Discount", fontsize=9, color="#006600", ha="center", va="center", alpha=0.8)
        if y_0618 is not None and y_0786 is not None:
            ax.text(mid_x, (y_0618 + y_0786) / 2, "OTE", fontsize=9, color="#886600", ha="center", va="center", alpha=0.9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=1))

    # Current price line & label
    ax.axhline(y=current_price, color="#1f77b4", linestyle=":", linewidth=1.0)
    ax.text(
        data_1d.index[int(len(data_1d) * 0.99)],
        current_price,
        f"Price {current_price:.2f}",
        fontsize=8,
        color="#1f77b4",
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
    )

    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    # Дополнительно сохраняем в output/ проекта для постоянного доступа
    proj_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    try:
        os.makedirs(proj_out_dir, exist_ok=True)
        # Сохраняем ту же фигуру в проектный output
        fig2, axlist2 = mpf.plot(
            data_1d,
            type="candle",
            style="yahoo",
            addplot=ap,
            figscale=1.2,
            volume=False,
            returnfig=True,
        )
        ax2 = axlist2[0]
        # Повтор наложений (короче, без подписей зон, но с линиями и ценой)
        for k, y in fib1d.key_levels.items():
            if len(x_right) >= 2:
                ax2.plot(x_right, [y] * len(x_right), color="#666", linestyle="--", linewidth=0.9, alpha=0.8)
        ax2.axhline(y=current_price, color="#1f77b4", linestyle=":", linewidth=1.0)
        fig2.savefig(os.path.join(proj_out_dir, "xauusd_swings_1d_candles.png"), dpi=130)
        plt.close(fig2)
    except Exception:
        pass
    assert out_path.exists(), "Картинка не сохранена"


