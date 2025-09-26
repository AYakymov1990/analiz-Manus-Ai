from datetime import datetime

import pytest

from src.core.data_structures import (
    StructureDirection,
    FibonacciZone,
    SetupType,
    SwingPoint,
    StructureAnalysis,
    FibonacciAnalysis,
    SupportResistanceLevel,
    LiquidityZone,
    SetupAnalysis,
    ManipulationStatus,
)


def test_enums_values():
    assert StructureDirection.BULLISH.value == "bullish"
    assert StructureDirection.BEARISH.value == "bearish"
    assert StructureDirection.SIDEWAYS.value == "sideways"

    assert FibonacciZone.PREMIUM.value == "premium"
    assert FibonacciZone.DISCOUNT.value == "discount"
    assert FibonacciZone.OTE.value == "ote"
    assert FibonacciZone.EQUILIBRIUM.value == "equilibrium"

    assert SetupType.TREND_CONTINUATION.value == "trend_continuation"
    assert SetupType.COUNTER_TREND_SSL_HUNT.value == "counter_trend_ssl_hunt"
    assert SetupType.STRUCTURE_BREAK.value == "structure_break"
    assert SetupType.NO_SETUP.value == "no_setup"


def test_swing_point_dataclass():
    ts = datetime(2024, 1, 1, 12, 0)
    sp = SwingPoint(timestamp=ts, price=1.2345, type="high", timeframe="1D", strength=0.8)

    assert sp.timestamp == ts
    assert sp.price == 1.2345
    assert sp.type == "high"
    assert sp.timeframe == "1D"
    assert 0 <= sp.strength <= 1


def test_structure_analysis_dataclass():
    ts = datetime(2024, 1, 1, 12, 0)
    high = SwingPoint(timestamp=ts, price=1.3, type="high", timeframe="1D", strength=0.7)
    low = SwingPoint(timestamp=ts, price=1.1, type="low", timeframe="1D", strength=0.6)

    sa = StructureAnalysis(
        timeframe="1D",
        direction=StructureDirection.BULLISH,
        last_swing_high=high,
        last_swing_low=low,
        structure_strength=0.75,
        break_level=1.31,
        confidence=0.8,
    )

    assert sa.timeframe == "1D"
    assert sa.direction is StructureDirection.BULLISH
    assert sa.last_swing_high.price == 1.3
    assert sa.last_swing_low.price == 1.1
    assert 0 <= sa.structure_strength <= 1
    assert sa.break_level == 1.31
    assert 0 <= sa.confidence <= 1


def test_fibonacci_analysis_dataclass():
    fa = FibonacciAnalysis(
        timeframe="H4",
        swing_high=1.4,
        swing_low=1.2,
        current_zone=FibonacciZone.PREMIUM,
        retracement_level=0.618,
        key_levels={"0.5": 1.3, "0.618": 1.309, "0.786": 1.326},
        ote_consolidation=True,
    )

    assert fa.timeframe == "H4"
    assert fa.swing_high == 1.4
    assert fa.swing_low == 1.2
    assert fa.current_zone is FibonacciZone.PREMIUM
    assert 0 <= fa.retracement_level <= 1
    assert "0.618" in fa.key_levels
    assert fa.ote_consolidation is True


def test_support_resistance_level_dataclass():
    level = SupportResistanceLevel(
        price=1.25,
        touches=4,
        strength=0.7,
        level_type="resistance",
        retail_likely_to_trade=True,
    )

    assert level.price == 1.25
    assert level.touches == 4
    assert 0 <= level.strength <= 1
    assert level.level_type in ("support", "resistance")
    assert isinstance(level.retail_likely_to_trade, bool)


def test_liquidity_zone_dataclass():
    zone = LiquidityZone(
        price=1.245,
        zone_type="SSL",
        strength=0.6,
        estimated_volume="medium",
        retail_logic="Stops under equal lows",
    )

    assert zone.price == 1.245
    assert zone.zone_type in ("SSL", "BSL")
    assert 0 <= zone.strength <= 1
    assert zone.estimated_volume in ("low", "medium", "high")
    assert isinstance(zone.retail_logic, str)


def test_setup_analysis_dataclass():
    setup = SetupAnalysis(
        setup_type=SetupType.COUNTER_TREND_SSL_HUNT,
        trade_direction="long",
        confidence=0.65,
        entry_conditions={"m15_manipulation": True},
        risk_reward=1.8,
    )

    assert setup.setup_type is SetupType.COUNTER_TREND_SSL_HUNT
    assert setup.trade_direction in ("long", "short")
    assert 0 <= setup.confidence <= 1
    assert isinstance(setup.entry_conditions, dict)
    assert setup.risk_reward > 0


def test_manipulation_status_dataclass():
    status = ManipulationStatus(
        expected_direction="long",
        target_level=1.255,
        manipulation_detected=True,
        entry_trigger_level=1.248,
        current_phase="return_to_range",
    )

    assert status.expected_direction in ("long", "short")
    assert isinstance(status.target_level, float)
    assert isinstance(status.manipulation_detected, bool)
    assert isinstance(status.entry_trigger_level, float)
    assert isinstance(status.current_phase, str)


