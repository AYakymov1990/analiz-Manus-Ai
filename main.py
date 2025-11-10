import argparse
import json
import os
from typing import Dict

import pandas as pd

from src.core.strategy_engine import TradingAnalyzer
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer
from src.analyzers.fibonacci_analyzer import MultiTimeframeFibonacciAnalyzer
from src.analyzers.retail_analyzer import RetailBehaviorAnalyzer
from src.core.oanda_data_collector import OANDADataCollector


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Strategy Analyzer - Context Builder for Manus AI")
    parser.add_argument("symbol", type=str, help="Trading symbol (e.g., EURUSD, GBPUSD, XAUUSD, NAS100, SPX500)")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON output to file")
    parser.add_argument("--emit-summary", action="store_true", help="Also emit legacy real_chain_summary.json")
    parser.add_argument("--single-file", action="store_true", help="Write only output/real_chain/manus_ai_context.json and skip symbol+timestamp file")
    args = parser.parse_args()

    analyzer = TradingAnalyzer(api_key=os.getenv("OANDA_API_KEY"))
    # If --single-file is used, suppress engine's timestamped output
    context = analyzer.analyze(args.symbol, save_output=not (args.no_save or args.single_file))

    print(json.dumps({
        "symbol": context.get("metadata", {}).get("symbol"),
        "timestamp": context.get("metadata", {}).get("analysis_timestamp"),
        "setup": context.get("trading_setup", {}).get("type"),
        "manipulation": context.get("manipulation_context", {}).get("expected_manipulation"),
        "fib_1d_zone": context.get("fibonacci_analysis", {}).get("1d", {}).get("zone"),
    }, ensure_ascii=False))

    # Write a single stable file if requested
    if args.single_file and not args.no_save:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "real_chain"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "manus_ai_context.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)

    if args.emit_summary:
        # Generate real_chain_summary.json using OANDA data
        collector = OANDADataCollector(api_key=os.getenv("OANDA_API_KEY"))
        raw = collector.get_candles(args.symbol, "15m", count=1000)
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

        def _ser_lvl(lvl):
            obj = {"price": lvl.price, "touches": lvl.touches, "strength": lvl.strength, "type": lvl.level_type}
            if getattr(lvl, "timeframe", None) is not None:
                obj["timeframe"] = lvl.timeframe
            if getattr(lvl, "zone_boundaries", None) is not None:
                lo, hi = lvl.zone_boundaries
                obj["zone_boundaries"] = [float(lo), float(hi)]
            if getattr(lvl, "obviousness_score", None) is not None:
                obj["obviousness_score"] = float(lvl.obviousness_score)
            if getattr(lvl, "last_touch", None) is not None:
                obj["last_touch"] = lvl.last_touch
            if getattr(lvl, "reaction_strengths", None) is not None:
                obj["reaction_strengths"] = [float(x) for x in (lvl.reaction_strengths or [])]
            if getattr(lvl, "time_separation_hours", None) is not None:
                obj["time_separation"] = [float(x) for x in (lvl.time_separation_hours or [])]
            return obj

        sr_1d = [_ser_lvl(lvl) for lvl in retail["support_resistance_levels"]]
        sr_4h = [_ser_lvl(lvl) for lvl in levels_4h]

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


