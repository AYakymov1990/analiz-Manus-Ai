import os
import sys
import json
import argparse
from typing import Optional, Tuple, List

import pandas as pd

# Ensure project root on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.strategy_engine import TradingAnalyzer  # type: ignore
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer  # type: ignore
from src.core.data_structures import SwingPoint  # type: ignore


def read_4h_boundaries_from_context(context_path: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Returns ((support_lower, support_upper), (resistance_lower, resistance_upper)) from manus_ai_context.json
    """
    if not os.path.exists(context_path):
        return None
    with open(context_path, "r", encoding="utf-8") as f:
        ctx = json.load(f)
    kb = ctx.get("retail_behavior", {}).get("key_sr_levels", {}).get("4H", {})
    sup = kb.get("support") or {}
    res = kb.get("resistance") or {}
    s_bounds = sup.get("zone_boundaries")
    r_bounds = res.get("zone_boundaries")
    if not (isinstance(s_bounds, list) and isinstance(r_bounds, list) and len(s_bounds) == 2 and len(r_bounds) == 2):
        return None
    return (float(s_bounds[0]), float(s_bounds[1])), (float(r_bounds[0]), float(r_bounds[1]))


def get_first_swing_high_above(swings: List[SwingPoint], threshold: float) -> Optional[SwingPoint]:
    candidates = [sp for sp in swings if sp.type == "high" and float(sp.price) > threshold]
    if not candidates:
        return None
    candidates.sort(key=lambda sp: sp.timestamp)
    return candidates[0]


def get_first_swing_low_below(swings: List[SwingPoint], threshold: float) -> Optional[SwingPoint]:
    candidates = [sp for sp in swings if sp.type == "low" and float(sp.price) < threshold]
    if not candidates:
        return None
    candidates.sort(key=lambda sp: sp.timestamp)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter 4H swings based on S/R boundaries")
    parser.add_argument("symbol", type=str, help="Symbol (e.g., MNQ1! or MNQ=F)")
    parser.add_argument("--context", type=str, default=os.path.join("output", "real_chain", "manus_ai_context.json"))
    args = parser.parse_args()

    # Load 4H boundaries from context
    boundaries = read_4h_boundaries_from_context(args.context)
    if not boundaries:
        print("Could not read 4H zone boundaries from context JSON.")
        sys.exit(1)
    (support_lower, support_upper), (resistance_lower, resistance_upper) = boundaries

    # Load data, with fallback from MNQ1! to MNQ=F if needed
    analyzer = TradingAnalyzer()
    symbol_used = args.symbol
    try:
        data = analyzer._load_and_validate_data(symbol_used)
    except Exception:
        if args.symbol.upper().startswith("MNQ"):
            symbol_used = "MNQ=F"
            data = analyzer._load_and_validate_data(symbol_used)
        else:
            raise

    df_4h: pd.DataFrame = data["4H"]
    current_close = float(df_4h["Close"].iloc[-1])

    struct_an = MultiTimeframeStructureAnalyzer(symbol_used)
    swings = struct_an.detect_swing_points(df_4h, "4H")

    first_high_above = get_first_swing_high_above(swings, resistance_upper)
    first_low_below = get_first_swing_low_below(swings, support_lower)

    print(f"Symbol used: {symbol_used}")
    print(f"Current close: {current_close:.5f}")
    print(f"4H resistance upper boundary: {resistance_upper:.5f}")
    print(f"4H support lower boundary   : {support_lower:.5f}")
    print("")

    if first_high_above:
        swept_now = current_close >= float(first_high_above.price)
        print("First swing HIGH above resistance:")
        print(f"  timestamp: {first_high_above.timestamp}")
        print(f"  price    : {float(first_high_above.price):.5f}")
        print(f"  strength : {float(first_high_above.strength):.3f}")
        print(f"  swept_now: {swept_now} (current_close >= swing.price)")
    else:
        print("No swing HIGH found above resistance upper boundary.")

    print("")

    if first_low_below:
        swept_now = current_close <= float(first_low_below.price)
        print("First swing LOW below support:")
        print(f"  timestamp: {first_low_below.timestamp}")
        print(f"  price    : {float(first_low_below.price):.5f}")
        print(f"  strength : {float(first_low_below.strength):.3f}")
        print(f"  swept_now: {swept_now} (current_close <= swing.price)")
    else:
        print("No swing LOW found below support lower boundary.")


if __name__ == "__main__":
    main()


