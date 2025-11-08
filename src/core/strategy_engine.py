from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

from ..analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from ..analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from ..analyzers.retail_analyzer import RetailBehaviorAnalyzer
from ..analyzers.setup_detector import SetupDetector
from ..analyzers.manipulation_detector import ManipulationDetector
from .context_builder import AnalysisResults, build_manus_context


class TradingAnalyzer:
    def __init__(self) -> None:
        self.structure_analyzer = MultiTimeframeStructureAnalyzer("*")
        self.fibonacci_analyzer = MultiTimeframeFibonacciAnalyzer()
        self.retail_analyzer = RetailBehaviorAnalyzer("*")
        self.setup_detector = SetupDetector()
        self.manipulation_detector = ManipulationDetector()

    def _ensure_ohlc(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df[symbol]
            except Exception:
                df = df.droplevel(-1, axis=1)
        return df[["Open", "High", "Low", "Close"]]

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        o = df["Open"].resample(rule).first()
        h = df["High"].resample(rule).max()
        l = df["Low"].resample(rule).min()
        c = df["Close"].resample(rule).last()
        res = pd.concat([o, h, l, c], axis=1).dropna()
        res.columns = ["Open", "High", "Low", "Close"]
        return res

    def _load_and_validate_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        assert yf is not None, "yfinance недоступен"
        raw = yf.download(symbol, period="59d", interval="15m", auto_adjust=False, progress=False, threads=False)
        if raw.empty:
            for iv in ("30m", "60m"):
                raw = yf.download(symbol, period="3mo", interval=iv, auto_adjust=False, progress=False, threads=False)
                if not raw.empty:
                    break
        if raw.empty:
            raise RuntimeError("Нет данных от yfinance")

        raw = self._ensure_ohlc(raw, symbol)
        raw.index = pd.to_datetime(raw.index)

        data_m15 = raw.copy()
        data_h4 = self._resample(raw, "4h")
        data_1d = self._resample(raw, "1D")

        # окна анализа
        win_1d = data_1d.loc[data_1d.index >= data_1d.index[-1] - pd.Timedelta(days=35)]
        win_4h = data_h4.loc[data_h4.index >= data_h4.index[-1] - pd.Timedelta(days=14)]

        return {"1D": win_1d, "4H": win_4h, "15M": data_m15}

    def analyze(self, symbol: str, save_output: bool = True) -> Dict[str, Any]:
        data = self._load_and_validate_data(symbol)
        current_price = float(data["15M"]["Close"].iloc[-1])

        structures = self.structure_analyzer.analyze_all_timeframes(data["1D"], data["4H"], data["15M"]) 
        fibs = self.fibonacci_analyzer.analyze_all_timeframes(structures, current_price)
        retail = self.retail_analyzer.analyze_retail_behavior(data["1D"], fibs, structures, data_h4=data["4H"])
        # Добавим 4H уровни S/R (SMC) в ретейл-контекст для приоритетного вывода
        try:
            retail["support_resistance_levels_h4"] = self.retail_analyzer.find_support_resistance_levels_h4(
                data["4H"], structures
            )
        except Exception:
            retail["support_resistance_levels_h4"] = []
        setup = self.setup_detector.detect_setup(structures, fibs, retail)
        manip = self.manipulation_detector.detect_manipulation(
            data["15M"].tail(500),
            {
                "current_price": current_price,
                "liquidity_zones": {
                    "BSL": [z.price for z in retail["liquidity_zones"] if z.zone_type == "BSL"],
                    "SSL": [z.price for z in retail["liquidity_zones"] if z.zone_type == "SSL"],
                },
                "fibonacci_zone": fibs["1D"].current_zone.value,
            },
        )

        analysis = AnalysisResults(
            symbol=symbol,
            current_price=current_price,
            data_windows={
                "1D": (str(data["1D"].index[0]), str(data["1D"].index[-1])),
                "4H": (str(data["4H"].index[0]), str(data["4H"].index[-1])),
            },
            structures=structures,
            fibonacci=fibs,
            retail=retail,
            setup=setup,
            manipulation=manip,
        )

        ctx = build_manus_context(symbol, analysis)

        if save_output:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "real_chain"))
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"manus_ai_context_{symbol}_{ts}.json")
            import json

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ctx, f, ensure_ascii=False, indent=2)

        return ctx


