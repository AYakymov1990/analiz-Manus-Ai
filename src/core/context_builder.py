from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .data_structures import (
    StructureAnalysis,
    FibonacciAnalysis,
    SupportResistanceLevel,
    LiquidityZone,
)


@dataclass
class AnalysisResults:
    symbol: str
    current_price: float
    data_windows: Dict[str, Tuple[str, str]]
    structures: Dict[str, StructureAnalysis]
    fibonacci: Dict[str, FibonacciAnalysis]
    retail: Dict[str, Any]
    setup: Any  # SetupResult-like (has setup_type, confidence, conditions)
    manipulation: Any  # ManipulationResult-like (has manipulation_type, confidence, details)


def _serialize_swing_point(sp: Any) -> Dict[str, Any]:
    return {
        "timestamp": sp.timestamp.isoformat() if hasattr(sp, "timestamp") else None,
        "price": float(getattr(sp, "price", None)),
        "type": getattr(sp, "type", None),
        "timeframe": getattr(sp, "timeframe", None),
        "strength": float(getattr(sp, "strength", 0.0)),
    }


def _serialize_structure(sa: StructureAnalysis) -> Dict[str, Any]:
    return {
        "direction": sa.direction.value,
        "strength": float(sa.structure_strength),
        "last_swing_high": _serialize_swing_point(sa.last_swing_high),
        "last_swing_low": _serialize_swing_point(sa.last_swing_low),
        "break_level": float(sa.break_level),
        "confidence": float(sa.confidence),
    }


def _serialize_fibonacci(fa: FibonacciAnalysis) -> Dict[str, Any]:
    return {
        "zone": fa.current_zone.value,
        "retracement": float(fa.retracement_level),
        "key_levels": {k: float(v) for k, v in fa.key_levels.items()},
        "swing_high": float(fa.swing_high),
        "swing_low": float(fa.swing_low),
    }


def _retail_implications_from_zone(zone_value: str) -> str:
    if zone_value.startswith("extension"):
        return "extension_context_risk_of_reversal"
    if zone_value == "premium":
        return "retail_unlikely_to_buy"
    if zone_value == "discount":
        return "retail_likely_to_buy"
    if zone_value == "ote":
        return "possible_consolidation_and_traps"
    return "neutral"


def _serialize_sr_level(lvl: SupportResistanceLevel) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "price": float(lvl.price),
        "touches": int(lvl.touches),
        "strength": float(lvl.strength),
        "type": lvl.level_type,
        "retail_likely": bool(lvl.retail_likely_to_trade),
    }
    # Optional SMC fields
    if getattr(lvl, "timeframe", None) is not None:
        obj["timeframe"] = lvl.timeframe
    if getattr(lvl, "zone_boundaries", None) is not None:
        lo, hi = lvl.zone_boundaries  # type: ignore
        obj["zone_boundaries"] = [float(lo), float(hi)]
    if getattr(lvl, "obviousness_score", None) is not None:
        obj["obviousness_score"] = float(lvl.obviousness_score)  # type: ignore
    if getattr(lvl, "last_touch", None) is not None:
        obj["last_touch"] = lvl.last_touch  # ISO str
    if getattr(lvl, "reaction_strengths", None) is not None:
        obj["reaction_strengths"] = [float(x) for x in (lvl.reaction_strengths or [])]
    # Map time_separation_hours -> time_separation
    if getattr(lvl, "time_separation_hours", None) is not None:
        obj["time_separation"] = [float(x) for x in (lvl.time_separation_hours or [])]
    return obj


def _serialize_liquidity_zone(z: LiquidityZone) -> Dict[str, Any]:
    return {
        "price": float(z.price),
        "type": z.zone_type,
        "strength": float(z.strength),
        "volume": z.estimated_volume,
        "logic": z.retail_logic,
    }


def _derive_sentiment(retail_entry_analysis: Dict[str, Any]) -> str:
    base = retail_entry_analysis.get("base_probability", "medium")
    likely = retail_entry_analysis.get("retail_likely_to_enter", False)
    if base == "high" or likely:
        return "bullish_retail_interest"
    if base == "low":
        return "bearish_retail_interest"
    return "neutral"


def build_manus_context(symbol: str, analysis_results: AnalysisResults) -> Dict[str, Any]:
    fib1d_zone = analysis_results.fibonacci["1D"].current_zone.value

    # Retail blocks
    sr_levels_h4 = analysis_results.retail.get("support_resistance_levels_h4", [])
    sr_levels = analysis_results.retail.get("support_resistance_levels", [])
    liq_zones = analysis_results.retail.get("liquidity_zones", [])
    retail_entry = analysis_results.retail.get("retail_entry_analysis", {})

    # Choose SSL target (min SSL) for question formatting
    ssl_targets = [z.price for z in liq_zones if getattr(z, "zone_type", "") == "SSL"]
    ssl_target = float(min(ssl_targets)) if ssl_targets else None

    manipulation = analysis_results.manipulation
    manip_type_val = getattr(getattr(manipulation, "manipulation_type", None), "value", None)

    extension_scenario = fib1d_zone if fib1d_zone.startswith("extension") else "non_extension"
    retail_sentiment = _derive_sentiment(retail_entry)

    context: Dict[str, Any] = {
        "metadata": {
            "symbol": symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_windows": analysis_results.data_windows,
            "validation_status": "all_swing_levels_validated_against_ohlc",
        },
        "market_structure": {
            "1d": _serialize_structure(analysis_results.structures["1D"]),
            "4h": _serialize_structure(analysis_results.structures["4H"]),
            "15m": _serialize_structure(analysis_results.structures["15M"]),
        },
        "fibonacci_analysis": {
            "1d": {
                **_serialize_fibonacci(analysis_results.fibonacci["1D"]),
                "retail_implications": _retail_implications_from_zone(fib1d_zone),
            },
            "4h": _serialize_fibonacci(analysis_results.fibonacci["4H"]),
            "15m": _serialize_fibonacci(analysis_results.fibonacci["15M"]),
        },
        "retail_behavior": {
            "sentiment": retail_sentiment,
            "support_resistance_4h": [_serialize_sr_level(x) for x in sr_levels_h4],
            "support_resistance_1d": [_serialize_sr_level(x) for x in sr_levels],
            "liquidity_zones": [_serialize_liquidity_zone(z) for z in liq_zones],
            "vulnerability_assessment": {
                "near_support": bool(retail_entry.get("nearby_support")),
                "retail_likely_to_enter": bool(retail_entry.get("retail_likely_to_enter", False)),
            },
        },
        "trading_setup": {
            "type": analysis_results.setup.setup_type.value,
            "confidence": float(analysis_results.setup.confidence),
            "conditions": getattr(analysis_results.setup, "conditions", {}),
            "risk_reward": getattr(analysis_results.setup, "risk_reward", None),
            "entry_criteria": getattr(analysis_results.setup, "conditions", {}),
        },
        "manipulation_context": {
            "recent_manipulations": [manip_type_val] if manip_type_val else [],
            "expected_manipulation": manip_type_val,
            "timing_signals": getattr(analysis_results.manipulation, "details", {}),
        },
        "manus_ai_questions": [
            f"Should we execute counter-trend SSL hunt to {ssl_target} or wait for trend continuation?",
            f"What is the optimal entry timing given current {fib1d_zone} and {manip_type_val}?",
            f"How should we manage risk in this {extension_scenario} with {retail_sentiment}?",
            "What is the primary target: SSL hunt or structure continuation?",
        ],
    }

    return context


