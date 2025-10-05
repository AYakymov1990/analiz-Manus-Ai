import argparse
import json
import os
from typing import Dict

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

from src.core.strategy_engine import TradingAnalyzer
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.analyzers.retail_analyzer import RetailBehaviorAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Strategy Analyzer - Context Builder for Manus AI")
    parser.add_argument("symbol", type=str, help="Yahoo Finance symbol, e.g., GC=F, EURUSD=X, BTC-USD")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON output to file")
    parser.add_argument("--emit-summary", action="store_true", help="Also emit legacy real_chain_summary.json")
    args = parser.parse_args()

    analyzer = TradingAnalyzer()
    context = analyzer.analyze(args.symbol, save_output=not args.no_save)

    print(json.dumps({
        "symbol": context.get("metadata", {}).get("symbol"),
        "timestamp": context.get("metadata", {}).get("analysis_timestamp"),
        "setup": context.get("trading_setup", {}).get("type"),
        "manipulation": context.get("manipulation_context", {}).get("expected_manipulation"),
        "fib_1d_zone": context.get("fibonacci_analysis", {}).get("1d", {}).get("zone"),
    }, ensure_ascii=False))

    if args.emit_summary:
        assert yf is not None, "yfinance недоступен"
        # Сгенерировать real_chain_summary.json (совместимо с тестом)
        raw = yf.download(args.symbol, period="59d", interval="15m", auto_adjust=False, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                raw = raw[args.symbol]
            except Exception:
                raw = raw.droplevel(-1, axis=1)
        raw = raw[["Open", "High", "Low", "Close"]]
        raw.index = pd.to_datetime(raw.index)
        data_m15 = raw.copy()
        data_h4 = pd.concat([
            raw["Open"].resample("4h").first(),
            raw["High"].resample("4h").max(),
            raw["Low"].resample("4h").min(),
            raw["Close"].resample("4h").last(),
        ], axis=1).dropna()
        data_h4.columns = ["Open", "High", "Low", "Close"]
        data_1d = pd.concat([
            raw["Open"].resample("1D").first(),
            raw["High"].resample("1D").max(),
            raw["Low"].resample("1D").min(),
            raw["Close"].resample("1D").last(),
        ], axis=1).dropna()
        data_1d.columns = ["Open", "High", "Low", "Close"]

        win_1d = data_1d.loc[data_1d.index >= data_1d.index[-1] - pd.Timedelta(days=35)]
        win_4h = data_h4.loc[data_h4.index >= data_h4.index[-1] - pd.Timedelta(days=14)]
        current_price = float(data_m15["Close"].iloc[-1])

        sa = MultiTimeframeStructureAnalyzer(args.symbol)
        structures = sa.analyze_all_timeframes(win_1d, win_4h, data_m15)
        fa = MultiTimeframeFibonacciAnalyzer()
        fibs = fa.analyze_all_timeframes(structures, current_price)
        ra = RetailBehaviorAnalyzer(args.symbol)
        retail = ra.analyze_retail_behavior(win_1d, fibs, structures)
        levels_4h = ra.find_support_resistance_levels_h4(win_4h, structures)

        sr_1d = [
            {"price": lvl.price, "touches": lvl.touches, "strength": lvl.strength, "type": lvl.level_type}
            for lvl in retail["support_resistance_levels"]
        ]
        sr_4h = [
            {"price": lvl.price, "touches": lvl.touches, "strength": lvl.strength, "type": lvl.level_type}
            for lvl in levels_4h
        ]

        report: Dict = {
            "symbol": args.symbol,
            "data_windows": {
                "1D": [str(win_1d.index[0]), str(win_1d.index[-1])],
                "4H": [str(win_4h.index[0]), str(win_4h.index[-1])],
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
        }

        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "real_chain"))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "real_chain_summary.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


