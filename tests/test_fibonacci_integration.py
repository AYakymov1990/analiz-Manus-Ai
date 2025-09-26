import os
from typing import Dict

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")
import yfinance as yf  # type: ignore

pytest.importorskip("mplfinance")
import mplfinance as mpf  # type: ignore

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.core.data_structures import FibonacciZone, StructureAnalysis


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    res = pd.concat([o, h, l, c], axis=1).dropna()
    res.columns = ["Open", "High", "Low", "Close"]
    return res


def _download(symbol: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period="59d",
        interval="15m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        for iv in ["30m", "60m"]:
            data = yf.download(
                symbol,
                period="3mo",
                interval=iv,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not data.empty:
                break
    return data


def _ensure_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[symbol]
        except Exception:
            df = df.droplevel(-1, axis=1)
    return df[["Open", "High", "Low", "Close"]]


def _plot_fib(
    data: pd.DataFrame,
    fib_levels: Dict[str, float],
    swings: pd.DataFrame,
    current_price: float,
    out_path: str,
):
    swing_high_series = pd.Series(index=data.index, dtype=float)
    swing_low_series = pd.Series(index=data.index, dtype=float)
    for idx, row in swings.iterrows():
        if row["type"] == "high" and idx in swing_high_series.index:
            swing_high_series.loc[idx] = row["price"]
        elif row["type"] == "low" and idx in swing_low_series.index:
            swing_low_series.loc[idx] = row["price"]

    ap = []
    if swing_high_series.notna().any():
        ap.append(mpf.make_addplot(swing_high_series, type="scatter", markersize=30, marker="^", color="red"))
    if swing_low_series.notna().any():
        ap.append(mpf.make_addplot(swing_low_series, type="scatter", markersize=30, marker="v", color="green"))

    fig, axlist = mpf.plot(
        data,
        type="candle",
        style="yahoo",
        addplot=ap,
        figscale=1.2,
        volume=False,
        returnfig=True,
    )
    ax = axlist[0]

    # Горизонтальные линии уровней Фибо по всей ширине + подписи слева
    for k, y in fib_levels.items():
        ax.axhline(y=y, color="#888", linestyle="--", linewidth=0.8)
        ax.text(data.index[0], y, k, fontsize=8, color="#444", va="bottom")

    # Заливки зон по всей ширине
    y_0_5 = fib_levels.get("0.5")
    y_0618 = fib_levels.get("0.618")
    y_0786 = fib_levels.get("0.786")
    if y_0_5 is not None:
        ax.axhspan(ax.get_ylim()[0], y_0_5, facecolor="#c3f3c3", alpha=0.2)
        ax.axhspan(y_0_5, ax.get_ylim()[1], facecolor="#f8c3c3", alpha=0.12)
    if y_0618 is not None and y_0786 is not None:
        ax.axhspan(y_0618, y_0786, facecolor="#fff3b0", alpha=0.25)

    # Текущая цена (горизонтальная линия + подпись)
    ax.axhline(y=current_price, color="#1f77b4", linestyle=":", linewidth=1.0)
    ax.text(
        data.index[int(len(data) * 0.99)],
        current_price,
        f"Price {current_price:.2f}",
        fontsize=8,
        color="#1f77b4",
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
    )

    # Легенда для свингов
    labels = []
    if swings[swings["type"] == "high"].shape[0] > 0:
        labels.append("swing high")
    if swings[swings["type"] == "low"].shape[0] > 0:
        labels.append("swing low")
    if labels:
        ax.legend(labels, loc="upper left", fontsize=8)

    fig.savefig(out_path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Интеграционный тест требует сети; включить с RUN_INTEGRATION_TESTS=1",
)
def test_fibonacci_multi_timeframe_visual_and_summary(tmp_path):
    symbol = "GC=F"
    raw = _download(symbol)
    assert not raw.empty, "Нет данных от yfinance"

    raw = _ensure_columns(raw, symbol)
    raw.index = pd.to_datetime(raw.index)
    data_m15 = raw.copy()
    data_h4 = _resample(raw, "4h")
    data_1d = _resample(raw, "1D")

    current_price = float(data_m15["Close"].iloc[-1])

    # Структуры из Этапа 2
    sa = MultiTimeframeStructureAnalyzer(symbol)
    structures = sa.analyze_all_timeframes(data_1d, data_h4, data_m15)

    # Фибо-анализ
    fa = MultiTimeframeFibonacciAnalyzer()
    fibs = fa.analyze_all_timeframes(structures, current_price)

    # Подготовка свингов для отрисовки 1D
    swings_1d = sa.detect_swing_points(data_1d, "1D")
    swings_df = pd.DataFrame(
        {"timestamp": [s.timestamp for s in swings_1d], "price": [s.price for s in swings_1d], "type": [s.type for s in swings_1d]}
    ).set_index("timestamp")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "fibonacci"))
    os.makedirs(out_dir, exist_ok=True)

    # 1D визуализация
    out_1d = os.path.join(out_dir, "fibonacci_1d_analysis.png")
    window_1d = data_1d.tail(120)
    swings_win = swings_df[swings_df.index.isin(window_1d.index)]
    _plot_fib(
        window_1d,
        fibs["1D"].key_levels,
        swings_win,
        current_price,
        out_1d,
    )

    # H4 визуализация (последние ~10 дней)
    out_h4 = os.path.join(out_dir, "fibonacci_h4_analysis.png")
    _plot_fib(
        data_h4.tail(60),
        fibs["4H"].key_levels,
        pd.DataFrame(columns=["price", "type"]).reindex(data_h4.tail(60).index),
        current_price,
        out_h4,
    )

    # Мульти-таймфрейм краткое резюме (как изображение с текстом)
    out_summary = os.path.join(out_dir, "fibonacci_multi_timeframe.png")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    lines = [
        "=== MULTI-TIMEFRAME SUMMARY ===",
        f"1D: {fibs['1D'].current_zone.value} (retracement={fibs['1D'].retracement_level:.3f})",
        f"4H: {fibs['4H'].current_zone.value} (retracement={fibs['4H'].retracement_level:.3f})",
        f"15M: {fibs['15M'].current_zone.value} (retracement={fibs['15M'].retracement_level:.3f})",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.savefig(out_summary, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Консольный вывод (валидируем как smoke)
    print("\n=== FIBONACCI ANALYSIS RESULTS ===")
    for tf in ["1D", "4H", "15M"]:
        f = fibs[tf]
        rng = abs(f.swing_high - f.swing_low)
        print(f"{tf} Analysis: range={rng:.2f}, retracement={f.retracement_level:.3f}, zone={f.current_zone.value}")

    # Проверки наличия файлов и типов
    assert os.path.exists(out_1d)
    assert os.path.exists(out_h4)
    assert os.path.exists(out_summary)


