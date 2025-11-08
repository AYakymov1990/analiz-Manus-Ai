from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.data_structures import (
    FibonacciAnalysis,
    FibonacciZone,
    LiquidityZone,
    StructureAnalysis,
    SupportResistanceLevel,
)
from .smc_sr_detector import SmartMoneySRDetector


class RetailBehaviorAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.min_touches = 3

    def analyze_retail_behavior(
        self,
        data_1d: pd.DataFrame,
        fibonacci: Dict[str, FibonacciAnalysis],
        structures: Dict[str, StructureAnalysis],
    ) -> Dict:
        sr_levels = self.find_support_resistance_levels(data_1d)
        sr_levels = self._inject_structure_levels(sr_levels, structures, fibonacci.get("1D"))

        retail_entry_analysis = self.analyze_retail_entry_probability(
            fibonacci["1D"], sr_levels
        )

        liquidity_zones = self.identify_liquidity_zones(structures, sr_levels)

        return {
            "support_resistance_levels": sr_levels,
            "retail_entry_analysis": retail_entry_analysis,
            "liquidity_zones": liquidity_zones,
        }

    def _inject_structure_levels(
        self,
        sr_levels: List[SupportResistanceLevel],
        structures: Dict[str, StructureAnalysis],
        fib_1d: Optional[FibonacciAnalysis] = None,
    ) -> List[SupportResistanceLevel]:
        levels = list(sr_levels)
        d1 = structures.get("1D")
        if d1 is not None:
            current_price = None
            if fib_1d is not None:
                try:
                    current_price = float(
                        fib_1d.swing_low + (fib_1d.swing_high - fib_1d.swing_low) * fib_1d.retracement_level
                    )
                except Exception:
                    current_price = None
            anchors: List[SupportResistanceLevel] = [
                SupportResistanceLevel(
                    price=float(d1.last_swing_low.price),
                    touches=max(self.min_touches, 3),
                    strength=0.8,
                    level_type="support",
                    retail_likely_to_trade=True,
                    timeframe="1D",
                    zone_boundaries=(
                        float(d1.last_swing_low.price) * 0.999,
                        float(d1.last_swing_low.price) * 1.001,
                    ),
                    obviousness_score=0.7,
                    touch_timestamps=[getattr(d1.last_swing_low, "timestamp", None).isoformat() if getattr(d1.last_swing_low, "timestamp", None) else None],
                    last_touch=getattr(d1.last_swing_low, "timestamp", None).isoformat() if getattr(d1.last_swing_low, "timestamp", None) else None,
                    distance_percent=(
                        abs(float(d1.last_swing_low.price) - current_price) / max(current_price, 1e-9)
                        if current_price is not None
                        else None
                    ),
                ),
                SupportResistanceLevel(
                    price=float(d1.last_swing_high.price),
                    touches=max(self.min_touches, 3),
                    strength=0.8,
                    level_type="resistance",
                    retail_likely_to_trade=True,
                    timeframe="1D",
                    zone_boundaries=(
                        float(d1.last_swing_high.price) * 0.999,
                        float(d1.last_swing_high.price) * 1.001,
                    ),
                    obviousness_score=0.7,
                    touch_timestamps=[getattr(d1.last_swing_high, "timestamp", None).isoformat() if getattr(d1.last_swing_high, "timestamp", None) else None],
                    last_touch=getattr(d1.last_swing_high, "timestamp", None).isoformat() if getattr(d1.last_swing_high, "timestamp", None) else None,
                    distance_percent=(
                        abs(float(d1.last_swing_high.price) - current_price) / max(current_price, 1e-9)
                        if current_price is not None
                        else None
                    ),
                ),
            ]
            for a in anchors:
                if not any(abs(a.price - x.price) / max(a.price, 1e-9) < 0.001 and a.level_type == x.level_type for x in levels):
                    levels.append(a)
        return levels

    def find_support_resistance_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        potential_supports: List[float] = []
        potential_resistances: List[float] = []

        if len(data) < 5:
            return []

        high = data["High"].astype(float)
        low = data["Low"].astype(float)
        close = data["Close"].astype(float)
        tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
        atr = tr.rolling(window=14, min_periods=1).mean().fillna(tr.expanding().mean())
        price_tolerance = (atr * 0.5).fillna(0.0)

        for i in range(2, len(data) - 2):
            low_i = float(data.iloc[i]["Low"]) 
            if (
                low_i < float(data.iloc[i - 1]["Low"]) 
                and low_i < float(data.iloc[i - 2]["Low"]) 
                and low_i < float(data.iloc[i + 1]["Low"]) 
                and low_i < float(data.iloc[i + 2]["Low"]) 
            ):
                if (float(data.iloc[i - 1]["Low"]) - low_i) > float(price_tolerance.iloc[i]) or (
                    float(data.iloc[i + 1]["Low"]) - low_i
                ) > float(price_tolerance.iloc[i]):
                    potential_supports.append(low_i)

        for i in range(2, len(data) - 2):
            high_i = float(data.iloc[i]["High"]) 
            if (
                high_i > float(data.iloc[i - 1]["High"]) 
                and high_i > float(data.iloc[i - 2]["High"]) 
                and high_i > float(data.iloc[i + 1]["High"]) 
                and high_i > float(data.iloc[i + 2]["High"]) 
            ):
                if (high_i - float(data.iloc[i - 1]["High"])) > float(price_tolerance.iloc[i]) or (
                    high_i - float(data.iloc[i + 1]["High"]) 
                ) > float(price_tolerance.iloc[i]):
                    potential_resistances.append(high_i)

        support_levels = self._validate_sr_levels(potential_supports, data, "support")
        resistance_levels = self._validate_sr_levels(potential_resistances, data, "resistance")
        return support_levels + resistance_levels

    def find_support_resistance_levels_h4(self, data_h4: pd.DataFrame, structures: Dict[str, StructureAnalysis]) -> List[SupportResistanceLevel]:
        """S/R для 4H по SMC с обязательной инъекцией якорей (минимум 3 касания)."""
        detector = SmartMoneySRDetector(
            lookback_candles=300,
            min_touches=max(self.min_touches, 3),
            max_touches=10,
            tolerance_percent=0.001,
            min_reaction_percent=0.002,
            min_time_separation_hours=12.0,
            max_levels=6,
            pivot_span=2,
        )
        levels = detector.detect(data_h4, timeframe="4H")
        h4 = structures.get("4H")
        if h4 is None:
            return levels
        # Текущее значение цены H4 для distance_percent
        try:
            curr_price_h4 = float(data_h4["Close"].iloc[-1])
        except Exception:
            curr_price_h4 = None
        anchors: List[SupportResistanceLevel] = [
            SupportResistanceLevel(
                price=float(h4.last_swing_low.price),
                touches=max(self.min_touches, 3),
                strength=0.7,
                level_type="support",
                retail_likely_to_trade=True,
                timeframe="4H",
                zone_boundaries=(
                    float(h4.last_swing_low.price) * 0.999,
                    float(h4.last_swing_low.price) * 1.001,
                ),
                obviousness_score=0.7,
                touch_timestamps=[getattr(h4.last_swing_low, "timestamp", None).isoformat() if getattr(h4.last_swing_low, "timestamp", None) else None],
                last_touch=getattr(h4.last_swing_low, "timestamp", None).isoformat() if getattr(h4.last_swing_low, "timestamp", None) else None,
                distance_percent=(
                    abs(float(h4.last_swing_low.price) - curr_price_h4) / max(curr_price_h4, 1e-9)
                    if curr_price_h4 is not None
                    else None
                ),
            ),
            SupportResistanceLevel(
                price=float(h4.last_swing_high.price),
                touches=max(self.min_touches, 3),
                strength=0.7,
                level_type="resistance",
                retail_likely_to_trade=True,
                timeframe="4H",
                zone_boundaries=(
                    float(h4.last_swing_high.price) * 0.999,
                    float(h4.last_swing_high.price) * 1.001,
                ),
                obviousness_score=0.7,
                touch_timestamps=[getattr(h4.last_swing_high, "timestamp", None).isoformat() if getattr(h4.last_swing_high, "timestamp", None) else None],
                last_touch=getattr(h4.last_swing_high, "timestamp", None).isoformat() if getattr(h4.last_swing_high, "timestamp", None) else None,
                distance_percent=(
                    abs(float(h4.last_swing_high.price) - curr_price_h4) / max(curr_price_h4, 1e-9)
                    if curr_price_h4 is not None
                    else None
                ),
            ),
        ]
        def _present(levels_list: List[SupportResistanceLevel], a: SupportResistanceLevel, tol: float = 0.001) -> bool:
            return any(abs(a.price - x.price) / max(a.price, 1e-9) < tol and a.level_type == x.level_type for x in levels_list)
        for a in anchors:
            if not _present(levels, a):
                levels.append(a)
        return levels

    def _validate_sr_levels(
        self, potential_levels: List[float], data: pd.DataFrame, level_type: str
    ) -> List[SupportResistanceLevel]:
        if not potential_levels:
            return []

        grouped_levels: List[List[float]] = []
        sorted_levels = sorted(potential_levels)
        current_group = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if abs(level - current_group[-1]) / max(current_group[-1], 1e-9) < 0.002:
                current_group.append(level)
            else:
                grouped_levels.append(current_group)
                current_group = [level]
        grouped_levels.append(current_group)

        sr_levels: List[SupportResistanceLevel] = []
        for group in grouped_levels:
            avg_price = float(np.mean(group))
            touches = int(len(group))
            if touches >= self.min_touches:
                strength = min(touches / 5.0, 1.0)
                retail_likely = self._assess_retail_likelihood(avg_price, level_type)
                sr_levels.append(
                    SupportResistanceLevel(
                        price=avg_price,
                        touches=touches,
                        strength=strength,
                        level_type=level_type,
                        retail_likely_to_trade=retail_likely,
                    )
                )
        return sr_levels

    def _assess_retail_likelihood(self, level_price: float, level_type: str) -> bool:
        return level_type == "support"

    def analyze_retail_entry_probability(
        self, fib_1d: FibonacciAnalysis, sr_levels: List[SupportResistanceLevel]
    ) -> Dict:
        zone = fib_1d.current_zone
        if zone == FibonacciZone.DISCOUNT:
            base_probability = "high"
            reasoning = "Price in discount zone - retail want to buy at 'cheaper' prices"
        elif zone == FibonacciZone.PREMIUM:
            base_probability = "low"
            reasoning = "Price in premium zone - retail hesitant to buy at 'expensive' prices"
        elif zone == FibonacciZone.OTE:
            base_probability = "medium"
            reasoning = "Price in OTE zone - some retail may enter"
        elif zone in (FibonacciZone.EXTENSION_ABOVE, FibonacciZone.EXTENSION_BELOW):
            base_probability = "low"
            reasoning = "Price in extension zone - retail tend to chase, lower quality entries"
        else:
            base_probability = "medium"
            reasoning = "Price at equilibrium"

        current_price = fib_1d.swing_low + (fib_1d.swing_high - fib_1d.swing_low) * fib_1d.retracement_level
        nearby_support = None
        for level in sr_levels:
            if level.level_type == "support" and abs(current_price - level.price) / max(current_price, 1e-9) < 0.005:
                nearby_support = level
                break

        return {
            "base_probability": base_probability,
            "reasoning": reasoning,
            "nearby_support": nearby_support,
            "retail_likely_to_enter": base_probability == "high" or nearby_support is not None,
        }

    def identify_liquidity_zones(
        self, structures: Dict[str, StructureAnalysis], sr_levels: List[SupportResistanceLevel]
    ) -> List[LiquidityZone]:
        liquidity_zones: List[LiquidityZone] = []
        for timeframe, structure in structures.items():
            if timeframe in ["1D", "4H"]:
                ssl_price = float(structure.last_swing_low.price) * 0.999
                liquidity_zones.append(
                    LiquidityZone(
                        price=ssl_price,
                        zone_type="SSL",
                        strength=structure.structure_strength,
                        estimated_volume="high" if timeframe == "1D" else "medium",
                        retail_logic=f"Retail stops placed below {timeframe} swing low at {structure.last_swing_low.price}",
                    )
                )
        for timeframe, structure in structures.items():
            if timeframe in ["1D", "4H"]:
                bsl_price = float(structure.last_swing_high.price) * 1.001
                liquidity_zones.append(
                    LiquidityZone(
                        price=bsl_price,
                        zone_type="BSL",
                        strength=structure.structure_strength,
                        estimated_volume="high" if timeframe == "1D" else "medium",
                        retail_logic=f"Retail stops placed above {timeframe} swing high at {structure.last_swing_high.price}",
                    )
                )
        return liquidity_zones


