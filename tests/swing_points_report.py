import os
import argparse
from typing import List
import sys

import pandas as pd

# Ensure project root on sys.path for `src` imports
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.strategy_engine import TradingAnalyzer  # type: ignore
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer  # type: ignore
from src.core.data_structures import SwingPoint  # type: ignore


def swings_to_rows(swings: List[SwingPoint], timeframe: str) -> List[dict]:
    rows: List[dict] = []
    for sp in swings:
        rows.append(
            {
                "timeframe": timeframe,
                "timestamp": getattr(sp, "timestamp", None),
                "type": getattr(sp, "type", ""),
                "price": float(getattr(sp, "price", 0.0) or 0.0),
                "strength": float(getattr(sp, "strength", 0.0) or 0.0),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate swing points report for a symbol")
    parser.add_argument("symbol", type=str, help="Symbol, e.g., EURUSD=X or MNQ=F")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("output", "real_chain"),
        help="Output directory for CSV/MD reports",
    )
    args = parser.parse_args()

    analyzer = TradingAnalyzer()
    data = analyzer._load_and_validate_data(args.symbol)

    struct_an = MultiTimeframeStructureAnalyzer(args.symbol)

    all_rows: List[dict] = []
    for tf in ("1D", "4H", "15M"):
        df = data.get(tf)
        if df is None or df.empty:
            continue
        swings = struct_an.detect_swing_points(df, tf)
        all_rows.extend(swings_to_rows(swings, tf))

    if not all_rows:
        print("No swings found.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out.sort_values(by=["timeframe", "timestamp"], inplace=True)

    os.makedirs(args.out_dir, exist_ok=True)
    base = f"swing_points_{args.symbol.replace('=', '').replace('^','').replace('/','_')}"
    csv_path = os.path.join(args.out_dir, f"{base}.csv")
    md_path = os.path.join(args.out_dir, f"{base}.md")

    df_out.to_csv(csv_path, index=False)

    # Save a simple markdown table (no external deps)
    cols = list(df_out.columns)
    header = "|" + "|".join(cols) + "|\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    lines = []
    for _, row in df_out.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if hasattr(v, "isoformat"):
                vals.append(str(v))
            else:
                vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Swing Points Report for {args.symbol}\n\n")
        f.write(header)
        f.write(sep)
        for ln in lines:
            f.write(ln)
        f.write("\n")

    # Print a compact sample to console
    print(df_out.head(20).to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved MD : {md_path}")


if __name__ == "__main__":
    main()


