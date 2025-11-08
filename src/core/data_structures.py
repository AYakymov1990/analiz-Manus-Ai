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
    EXTENSION_ABOVE = "extension_above"  # > 1.0
    EXTENSION_BELOW = "extension_below"  # < 0.0


class SetupType(Enum):
    TREND_CONTINUATION = "trend_continuation"
    COUNTER_TREND_SSL_HUNT = "counter_trend_ssl_hunt"
    STRUCTURE_BREAK = "structure_break"
    STRUCTURE_BREAK_WAIT = "structure_break_wait"
    EXTENSION_REVERSAL = "extension_reversal"
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
    # Полные списки свингов (для вычисления SSL/BSL относительно S&R зон)
    all_swing_highs: Optional[List[SwingPoint]] = None
    all_swing_lows: Optional[List[SwingPoint]] = None


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
    # Optional SMC metadata
    timeframe: Optional[str] = None
    zone_boundaries: Optional[Tuple[float, float]] = None
    obviousness_score: Optional[float] = None
    touch_timestamps: Optional[List[str]] = None
    last_touch: Optional[str] = None
    reaction_strengths: Optional[List[float]] = None
    time_separation_hours: Optional[List[float]] = None
    distance_percent: Optional[float] = None


@dataclass
class LiquidityZone:
    price: float
    zone_type: str  # "SSL" or "BSL"
    strength: float
    estimated_volume: str  # "low", "medium", "high"
    retail_logic: str
    timeframe: Optional[str] = None
    # Связь с источником (S&R и swing)
    derived_from_sr_price: Optional[float] = None
    derived_from_sr_boundaries: Optional[Tuple[float, float]] = None
    sr_timeframe: Optional[str] = None
    swing_timestamp: Optional[datetime] = None
    swing_strength: Optional[float] = None


@dataclass
class KeySRLevel:
    """
    Ключевой S&R уровень (без поля price, только границы зоны).
    """
    zone_boundaries: Tuple[float, float]
    strength: float
    obviousness_score: float
    touches: int
    last_touch: Optional[str] = None
    reaction_strengths: Optional[List[float]] = None
    time_separation_hours: Optional[List[float]] = None


@dataclass
class KeySRLevels:
    """
    Контейнер 4 ключевых уровней: 1D.support/resistance и 4H.support/resistance.
    """
    d1_support: Optional[KeySRLevel] = None
    d1_resistance: Optional[KeySRLevel] = None
    h4_support: Optional[KeySRLevel] = None
    h4_resistance: Optional[KeySRLevel] = None

    def to_dict(self) -> Dict[str, Dict[str, Optional[Dict]]]:
        return {
            "1D": {
                "support": self._level_to_dict(self.d1_support) if self.d1_support else None,
                "resistance": self._level_to_dict(self.d1_resistance) if self.d1_resistance else None,
            },
            "4H": {
                "support": self._level_to_dict(self.h4_support) if self.h4_support else None,
                "resistance": self._level_to_dict(self.h4_resistance) if self.h4_resistance else None,
            },
        }

    def _level_to_dict(self, level: KeySRLevel) -> Dict:
        return {
            "zone_boundaries": [float(level.zone_boundaries[0]), float(level.zone_boundaries[1])],
            "strength": float(level.strength),
            "obviousness_score": float(level.obviousness_score),
            "touches": int(level.touches),
            "last_touch": level.last_touch,
            "reaction_strengths": [float(x) for x in (level.reaction_strengths or [])] if level.reaction_strengths else None,
            "time_separation_hours": [float(x) for x in (level.time_separation_hours or [])] if level.time_separation_hours else None,
        }

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


class ManipulationType(Enum):
    STOP_HUNT_BELOW = "stop_hunt_below"
    STOP_HUNT_ABOVE = "stop_hunt_above"
    FALSE_BREAKOUT = "false_breakout"
    LIQUIDITY_GRAB = "liquidity_grab"
    NO_MANIPULATION = "no_manipulation"


