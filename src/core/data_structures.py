from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class StructureDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class FibonacciZone(Enum):
    PREMIUM = "premium"      # > 0.5
    DISCOUNT = "discount"    # < 0.5
    OTE = "ote"             # 0.618-0.786
    EQUILIBRIUM = "equilibrium"  # ~0.5


class SetupType(Enum):
    TREND_CONTINUATION = "trend_continuation"
    COUNTER_TREND_SSL_HUNT = "counter_trend_ssl_hunt"
    STRUCTURE_BREAK = "structure_break"
    NO_SETUP = "no_setup"


@dataclass
class SwingPoint:
    timestamp: datetime
    price: float
    type: str  # "high" or "low"
    timeframe: str
    strength: float  # 0-1


@dataclass
class StructureAnalysis:
    timeframe: str
    direction: StructureDirection
    last_swing_high: SwingPoint
    last_swing_low: SwingPoint
    structure_strength: float
    break_level: float
    confidence: float


@dataclass
class FibonacciAnalysis:
    timeframe: str
    swing_high: float
    swing_low: float
    current_zone: FibonacciZone
    retracement_level: float
    key_levels: Dict[str, float]
    ote_consolidation: bool = False


@dataclass
class SupportResistanceLevel:
    price: float
    touches: int
    strength: float
    level_type: str  # "support" or "resistance"
    retail_likely_to_trade: bool


@dataclass
class LiquidityZone:
    price: float
    zone_type: str  # "SSL" or "BSL"
    strength: float
    estimated_volume: str  # "low", "medium", "high"
    retail_logic: str


@dataclass
class SetupAnalysis:
    setup_type: SetupType
    trade_direction: str  # "long" or "short"
    confidence: float
    entry_conditions: Dict
    risk_reward: float


@dataclass
class ManipulationStatus:
    expected_direction: str
    target_level: float
    manipulation_detected: bool
    entry_trigger_level: float
    current_phase: str


