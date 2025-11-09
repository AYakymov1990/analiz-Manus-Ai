import os
import sys
import argparse
from typing import Optional, List, Tuple

import pandas as pd

# Ensure project root import
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.strategy_engine import TradingAnalyzer  # type: ignore
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer  # type: ignore
from src.core.data_structures import SwingPoint  # type: ignore


def first_swing_after_boundary(
    swings: List[SwingPoint], boundary: float, want_type: str, direction: str
) -> Optional[SwingPoint]:
    """
    Pick the first chronological swing after boundary.
    - want_type: 'high' or 'low'
    - direction: 'above' (price > boundary) or 'below' (price < boundary)
    """
    if want_type == "high" and direction == "above":
        candidates = [s for s in swings if s.type == "high" and float(s.price) > boundary]
    elif want_type == "low" and direction == "below":
        candidates = [s for s in swings if s.type == "low" and float(s.price) < boundary]
    else:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda s: s.timestamp)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find first 4H swings after given S/R boundaries")
    parser.add_argument("symbol", type=str, help="Symbol, e.g., MNQ1! or MNQ=F")
    parser.add_argument("--res_upper", type=float, default=25380.10475, help="4H resistance upper boundary")
    parser.add_argument("--sup_lower", type=float, default=25172.05275, help="4H support lower boundary")
    args = parser.parse_args()

    symbol_used = args.symbol
    analyzer = TradingAnalyzer()

    # Try loading symbol, fallback to MNQ=F if MNQ1! not found
    try:
        data = analyzer._load_and_validate_data(symbol_used)
    except Exception:
        if args.symbol.upper().startswith("MNQ") and args.symbol.endswith("!"):
            fallback = "MNQ=F"
            symbol_used = fallback
            data = analyzer._load_and_validate_data(symbol_used)
            print(f"[info] Fallback to {fallback} (data provider has no {args.symbol})")
        else:
            raise

    df_h4 = data.get("4H")
    if df_h4 is None or df_h4.empty:
        print("No 4H data available.")
        return

    current_price = float(df_h4["Close"].iloc[-1])
    struct = MultiTimeframeStructureAnalyzer(symbol_used)
    swings = struct.detect_swing_points(df_h4, "4H")

    # First swing high above resistance upper
    first_bsl = first_swing_after_boundary(swings, args.res_upper, want_type="high", direction="above")
    # First swing low below support lower
    first_ssl = first_swing_after_boundary(swings, args.sup_lower, want_type="low", direction="below")

    print(f"\nSymbol used: {symbol_used}")
    print(f"Current 4H close: {current_price:.5f}")
    print(f"4H resistance upper: {args.res_upper:.5f} | 4H support lower: {args.sup_lower:.5f}\n")

    if first_bsl:
        swept_bsl = current_price >= float(first_bsl.price)
        status_bsl = "INVALID (already swept by current price)" if swept_bsl else "OK"
        print(
            f"First swing HIGH above resistance: "
            f"{first_bsl.timestamp}  price={float(first_bsl.price):.5f}  strength={float(first_bsl.strength):.2f}  -> {status_bsl}"
        )
    else:
        print("First swing HIGH above resistance: NOT FOUND")

    if first_ssl:
        swept_ssl = current_price <= float(first_ssl.price)
        status_ssl = "INVALID (already swept by current price)" if swept_ssl else "OK"
        print(
            f"First swing LOW below support   : "
            f"{first_ssl.timestamp}  price={float(first_ssl.price):.5f}  strength={float(first_ssl.strength):.2f}  -> {status_ssl}"
        )
    else:
        print("First swing LOW below support   : NOT FOUND")

    print()


if __name__ == "__main__":
    main()


