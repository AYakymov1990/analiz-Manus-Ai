import os
import sys
import argparse
from typing import List

import pandas as pd
import mplfinance as mpf

# Ensure project root on path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.oanda_data_collector import OANDADataCollector  # type: ignore
from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer  # type: ignore
from src.core.data_structures import SwingPoint  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 4H swings on last N candles")
    parser.add_argument("symbol", type=str, help="Symbol (e.g., GBPUSD, EURUSD, XAUUSD)")
    parser.add_argument("--count", type=int, default=1000, help="Candles to fetch from OANDA (15m granularity base)")
    parser.add_argument("--window", type=int, default=300, help="Last N 4H candles to display")
    parser.add_argument("--out", type=str, default=os.path.join("output", "real_chain", "swings_4h_plot.png"))
    parser.add_argument("--use-context", action="store_true", help="Read 4H zones from manus_ai_context.json")
    parser.add_argument("--context-path", type=str, default=os.path.join("output", "real_chain", "manus_ai_context.json"))
    parser.add_argument("--no-plot", action="store_true", help="Do not render chart, only print filtered swings")
    # Distance filters in pips (FX)
    parser.add_argument("--min-ssl-zone-gap", type=float, default=10.0, help="Min pips below support.lower for SSL")
    parser.add_argument("--min-ssl-current-gap", type=float, default=10.0, help="Min pips below current price for SSL")
    parser.add_argument("--min-bsl-zone-gap", type=float, default=20.0, help="Min pips above resistance.upper for BSL")
    parser.add_argument("--min-bsl-current-gap", type=float, default=15.0, help="Min pips above current price for BSL")
    args = parser.parse_args()

    # Prepare output dir
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Fetch data (we can fetch directly 4h candles)
    api_key = os.getenv("OANDA_API_KEY")
    if not api_key:
        print("ERROR: OANDA_API_KEY is not set. Export it or create a .env.")
        sys.exit(1)
    collector = OANDADataCollector(api_key=api_key)
    df_4h = collector.get_candles(args.symbol, "4h", count=max(args.window, 600))
    df_4h = df_4h[["Open", "High", "Low", "Close"]]
    df_4h.index = pd.to_datetime(df_4h.index)
    if len(df_4h) == 0:
        print("No data returned.")
        sys.exit(1)

    # Trim to last window candles for visualization
    df_vis = df_4h.tail(args.window).copy()

    # Detect swings on full 4H data, then filter to window
    analyzer = MultiTimeframeStructureAnalyzer(args.symbol)
    swings_all: List[SwingPoint] = analyzer.detect_swing_points(df_4h, "4H")
    window_start = df_vis.index[0]
    window_end = df_vis.index[-1]
    swings_in_window = [sp for sp in swings_all if window_start <= sp.timestamp <= window_end]

    # Optionally read context zones (4H support/resistance)
    support_lower = None
    support_upper = None
    support_last_touch = None
    resistance_lower = None
    resistance_upper = None
    resistance_last_touch = None
    if args.use_context and os.path.exists(args.context_path):
        try:
            ctx = pd.read_json(args.context_path, typ="series")
            # Try new key_sr_levels format first
            # retail_behavior.key_sr_levels.4H.support / resistance
            rb = ctx.get("retail_behavior", {})
            if isinstance(rb, dict):
                ksr = rb.get("key_sr_levels") or rb.get("key_sr_levels".upper()) or {}
                h4 = None
                if isinstance(ksr, dict):
                    # Some dumps may use string keys
                    h4 = ksr.get("4H") or ksr.get("h4")
                if isinstance(h4, dict):
                    sup = h4.get("support")
                    res = h4.get("resistance")
                    if isinstance(sup, dict) and isinstance(sup.get("zone_boundaries", None), list):
                        support_lower, support_upper = float(sup["zone_boundaries"][0]), float(sup["zone_boundaries"][1])
                        support_last_touch = sup.get("last_touch")
                    if isinstance(res, dict) and isinstance(res.get("zone_boundaries", None), list):
                        resistance_lower, resistance_upper = float(res["zone_boundaries"][0]), float(res["zone_boundaries"][1])
                        resistance_last_touch = res.get("last_touch")
                # Fallback to old arrays support_resistance_4h
                if support_lower is None or resistance_upper is None:
                    sr4 = rb.get("support_resistance_4h") or rb.get("support_resistance_4H")
                    if isinstance(sr4, list) and len(sr4) >= 2:
                        sup_levels = [x for x in sr4 if x.get("type") == "support" and isinstance(x.get("zone_boundaries"), list)]
                        res_levels = [x for x in sr4 if x.get("type") == "resistance" and isinstance(x.get("zone_boundaries"), list)]
                        if sup_levels:
                            support_lower, support_upper = float(sup_levels[0]["zone_boundaries"][0]), float(sup_levels[0]["zone_boundaries"][1])
                            support_last_touch = sup_levels[0].get("last_touch")
                        if res_levels:
                            resistance_lower, resistance_upper = float(res_levels[0]["zone_boundaries"][0]), float(res_levels[0]["zone_boundaries"][1])
                            resistance_last_touch = res_levels[0].get("last_touch")
        except Exception as e:
            print(f"Context read failed: {e}")

    # Filter first suitable swings per rules:
    # - First swing low below support.lower, ts >= support.last_touch (if provided), current_price > swing.price
    # - First swing high above resistance.upper, ts >= resistance.last_touch (if provided), current_price < swing.price
    current_price = float(df_4h["Close"].iloc[-1])
    first_ssl = None
    first_bsl = None
    if support_lower is not None:
        # фильтруем только в пределах текущего окна визуализации
        candidates_lows: List[SwingPoint] = [
            sp for sp in swings_in_window
            if sp.type == "low" and float(sp.price) < float(support_lower) and current_price > float(sp.price)
        ]
        # отбрасываем слишком близкие по пипсам к зоне/текущей цене
        def _ssl_ok(sp: SwingPoint) -> bool:
            dist_zone_pips = (float(support_lower) - float(sp.price)) * 10000.0
            dist_curr_pips = (current_price - float(sp.price)) * 10000.0
            return dist_zone_pips >= args.min_ssl_zone_gap and dist_curr_pips >= args.min_ssl_current_gap
        candidates_lows = [sp for sp in candidates_lows if _ssl_ok(sp)]
        # берём самый близкий по времени (последний), а не самый ранний
        candidates_lows.sort(key=lambda sp: sp.timestamp, reverse=True)
        first_ssl = candidates_lows[0] if candidates_lows else None
    if resistance_upper is not None:
        candidates_highs: List[SwingPoint] = [
            sp for sp in swings_in_window
            if sp.type == "high" and float(sp.price) > float(resistance_upper) and current_price < float(sp.price)
        ]
        # отбрасываем слишком близкие по пипсам к зоне/текущей цене
        def _bsl_ok(sp: SwingPoint) -> bool:
            dist_zone_pips = (float(sp.price) - float(resistance_upper)) * 10000.0
            dist_curr_pips = (float(sp.price) - current_price) * 10000.0
            return dist_zone_pips >= args.min_bsl_zone_gap and dist_curr_pips >= args.min_bsl_current_gap
        candidates_highs = [sp for sp in candidates_highs if _bsl_ok(sp)]
        candidates_highs.sort(key=lambda sp: sp.timestamp, reverse=True)
        first_bsl = candidates_highs[0] if candidates_highs else None

    # Build additional plot art: scatter for highs and lows
    highs_x = []
    highs_y = []
    lows_x = []
    lows_y = []
    for sp in swings_in_window:
        if sp.type == "high":
            highs_x.append(sp.timestamp)
            highs_y.append(float(sp.price))
        elif sp.type == "low":
            lows_x.append(sp.timestamp)
            lows_y.append(float(sp.price))

    # Build addplots aligned to df_vis.index to avoid length mismatch
    apds = []
    if highs_x:
        # keep only timestamps in window index
        highs_pairs = [(ts, val) for ts, val in zip(highs_x, highs_y) if ts in df_vis.index]
        s_high = pd.Series(index=df_vis.index, dtype=float)
        if highs_pairs:
            for ts, val in highs_pairs:
                s_high.loc[ts] = val
        apds.append(mpf.make_addplot(s_high, type="scatter", marker="^", color="red"))
    if lows_x:
        lows_pairs = [(ts, val) for ts, val in zip(lows_x, lows_y) if ts in df_vis.index]
        s_low = pd.Series(index=df_vis.index, dtype=float)
        if lows_pairs:
            for ts, val in lows_pairs:
                s_low.loc[ts] = val
        apds.append(mpf.make_addplot(s_low, type="scatter", marker="v", color="green"))

    # Plot candlesticks with swings markers (optional)
    if not args.no_plot:
        mpf.plot(
            df_vis,
            type="candle",
            style="yahoo",
            addplot=apds if apds else None,
            title=f"{args.symbol} 4H | last {args.window} candles (swings marked)",
            savefig=dict(fname=args.out, dpi=140, bbox_inches="tight"),
        )

    # Print summary to console
    print(f"Symbol: {args.symbol}")
    print(f"Window: {str(window_start)} → {str(window_end)} | candles: {len(df_vis)}")
    print(f"Swings in window: {len(swings_in_window)} (highs={sum(1 for s in swings_in_window if s.type=='high')}, lows={sum(1 for s in swings_in_window if s.type=='low')})")
    # Show last few
    for sp in swings_in_window[-20:]:
        print(f"{sp.timestamp} | {sp.type.upper()} | {float(sp.price):.5f} | strength={float(sp.strength):.3f}")
    print(f"Saved plot: {args.out}")
    # Print only filtered candidates based on context
    if args.use_context:
        print("---- Filtered swings from context (4H) ----")
        if support_lower is not None:
            if first_ssl:
                print(f"SSL: {first_ssl.timestamp} | LOW | {float(first_ssl.price):.5f} | after zone [{support_lower:.5f},{(support_upper or 0.0):.5f}] | not swept (current={current_price:.5f}) | gaps: zone={(support_lower - float(first_ssl.price))*10000:.1f}p, curr={(current_price - float(first_ssl.price))*10000:.1f}p")
            else:
                print("SSL: not found (either none after last_touch/zone or already swept by current price)")
        if resistance_upper is not None:
            if first_bsl:
                print(f"BSL: {first_bsl.timestamp} | HIGH | {float(first_bsl.price):.5f} | after zone [{(resistance_lower or 0.0):.5f},{resistance_upper:.5f}] | not swept (current={current_price:.5f}) | gaps: zone={(float(first_bsl.price) - resistance_upper)*10000:.1f}p, curr={(float(first_bsl.price) - current_price)*10000:.1f}p")
            else:
                print("BSL: not found (either none after last_touch/zone or already swept by current price)")


if __name__ == "__main__":
    main()


