from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.data_structures import (
    FibonacciAnalysis,
    FibonacciZone,
    LiquidityZone,
    SwingPoint,
    StructureAnalysis,
    SupportResistanceLevel,
    KeySRLevel,
    KeySRLevels,
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
        data_h4: Optional[pd.DataFrame] = None,
    ) -> Dict:
        # 1) Собрать уровни для 1D
        sr_levels_1d = self.find_support_resistance_levels(data_1d)
        sr_levels_1d = self._inject_structure_levels(sr_levels_1d, structures, fibonacci.get("1D"))

        # 2) Собрать уровни для 4H (если есть данные)
        sr_levels_4h: List[SupportResistanceLevel] = []
        if data_h4 is not None:
            try:
                sr_levels_4h = self.find_support_resistance_levels_h4(data_h4, structures)
            except Exception:
                pass

        # 3) Выбрать 4 ключевых уровня
        current_price = self._get_current_price(structures)
        key_sr_levels = self.select_key_sr_levels(sr_levels_1d, sr_levels_4h, current_price)

        # 4) Retail entry анализ (можно использовать полный набор уровней)
        retail_entry_analysis = self.analyze_retail_entry_probability(
            fibonacci["1D"], sr_levels_1d + sr_levels_4h
        )

        # 5) SSL/BSL только от 4 ключевых уровней
        liquidity_zones = self.identify_liquidity_zones_from_key_levels(structures, key_sr_levels)

        return {
            "support_resistance_levels": sr_levels_1d,  # оставляем для обратной совместимости (1D)
            "support_resistance_levels_h4": sr_levels_4h,  # и 4H
            "key_sr_levels": key_sr_levels.to_dict(),
            "retail_entry_analysis": retail_entry_analysis,
            "liquidity_zones": liquidity_zones,
        }

    def select_key_sr_levels(
        self,
        sr_levels_1d: List[SupportResistanceLevel],
        sr_levels_4h: List[SupportResistanceLevel],
        current_price: float,
    ) -> KeySRLevels:
        supports_1d = [sr for sr in sr_levels_1d if sr.level_type == "support" and sr.zone_boundaries is not None]
        resistances_1d = [sr for sr in sr_levels_1d if sr.level_type == "resistance" and sr.zone_boundaries is not None]
        supports_4h = [sr for sr in sr_levels_4h if sr.level_type == "support" and sr.zone_boundaries is not None]
        resistances_4h = [sr for sr in sr_levels_4h if sr.level_type == "resistance" and sr.zone_boundaries is not None]

        d1_support = self._select_best_sr(supports_1d, current_price, is_support=True)
        d1_resistance = self._select_best_sr(resistances_1d, current_price, is_support=False)
        h4_support = self._select_best_sr(supports_4h, current_price, is_support=True)
        h4_resistance = self._select_best_sr(resistances_4h, current_price, is_support=False)

        return KeySRLevels(
            d1_support=self._convert_to_key_sr_level(d1_support) if d1_support else None,
            d1_resistance=self._convert_to_key_sr_level(d1_resistance) if d1_resistance else None,
            h4_support=self._convert_to_key_sr_level(h4_support) if h4_support else None,
            h4_resistance=self._convert_to_key_sr_level(h4_resistance) if h4_resistance else None,
        )

    def _select_best_sr(
        self,
        sr_list: List[SupportResistanceLevel],
        current_price: float,
        is_support: bool,
    ) -> Optional[SupportResistanceLevel]:
        if not sr_list:
            return None
        from datetime import datetime
        now = datetime.now()
        scored: List[tuple[float, SupportResistanceLevel]] = []
        for sr in sr_list:
            score = float(sr.strength) * float((sr.obviousness_score or 0.7))
            if sr.last_touch:
                try:
                    last_touch_dt = datetime.fromisoformat(sr.last_touch.replace("+00:00", ""))
                    days_ago = (now - last_touch_dt).days
                    freshness_weight = max(0.5, 1.0 - (days_ago / 14.0))
                    score *= freshness_weight
                except Exception:
                    pass
            if current_price > 0:
                distance = abs(float(sr.price) - current_price) / current_price
                distance_weight = 1.0 / (1.0 + distance * 10.0)
                score *= distance_weight
            scored.append((score, sr))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _convert_to_key_sr_level(self, sr: SupportResistanceLevel) -> KeySRLevel:
        zb = sr.zone_boundaries
        assert zb is not None, "zone_boundaries must be present for KeySRLevel"
        return KeySRLevel(
            zone_boundaries=(float(zb[0]), float(zb[1])),
            strength=float(sr.strength),
            obviousness_score=float(sr.obviousness_score or 0.7),
            touches=int(sr.touches),
            last_touch=sr.last_touch,
            reaction_strengths=[float(x) for x in (sr.reaction_strengths or [])] if sr.reaction_strengths else None,
            time_separation_hours=[float(x) for x in (sr.time_separation_hours or [])] if sr.time_separation_hours else None,
        )

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

    def identify_liquidity_zones_from_key_levels(
        self,
        structures: Dict[str, StructureAnalysis],
        key_sr_levels: KeySRLevels,
    ) -> List[LiquidityZone]:
        liquidity_zones: List[LiquidityZone] = []
        current_price = self._get_current_price(structures)

        # 1D support -> SSL
        if key_sr_levels.d1_support and "1D" in structures:
            struct = structures["1D"]
            zone = self._find_ssl_from_key_level(key_sr_levels.d1_support, struct.all_swing_lows or [], current_price, "1D")
            if zone:
                liquidity_zones.append(zone)
        # 1D resistance -> BSL
        if key_sr_levels.d1_resistance and "1D" in structures:
            struct = structures["1D"]
            zone = self._find_bsl_from_key_level(key_sr_levels.d1_resistance, struct.all_swing_highs or [], current_price, "1D")
            if zone:
                liquidity_zones.append(zone)
        # 4H support -> SSL
        if key_sr_levels.h4_support and "4H" in structures:
            struct = structures["4H"]
            zone = self._find_ssl_from_key_level(key_sr_levels.h4_support, struct.all_swing_lows or [], current_price, "4H")
            if zone:
                liquidity_zones.append(zone)
        # 4H resistance -> BSL
        if key_sr_levels.h4_resistance and "4H" in structures:
            struct = structures["4H"]
            zone = self._find_bsl_from_key_level(key_sr_levels.h4_resistance, struct.all_swing_highs or [], current_price, "4H")
            if zone:
                liquidity_zones.append(zone)

        return self._deduplicate_liquidity_zones(liquidity_zones)

    def _find_ssl_from_key_level(
        self,
        key_level: KeySRLevel,
        swing_lows: List[SwingPoint],
        current_price: float,
        timeframe: str,
    ) -> Optional[LiquidityZone]:
        lower = float(key_level.zone_boundaries[0])
        candidates = [sw for sw in swing_lows if float(sw.price) < lower]
        if not candidates:
            return None
        candidates.sort(key=lambda sw: abs(float(sw.price) - lower))
        max_distance = lower * 0.02
        for sw in candidates:
            distance = lower - float(sw.price)
            if distance > max_distance:
                continue
            if current_price > 0 and abs(float(sw.price) - current_price) / current_price > 0.15:
                continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="SSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"Retail stops below {timeframe} support (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                derived_from_sr_boundaries=(float(key_level.zone_boundaries[0]), float(key_level.zone_boundaries[1])),
                sr_timeframe=timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            )
        return None

    def _find_bsl_from_key_level(
        self,
        key_level: KeySRLevel,
        swing_highs: List[SwingPoint],
        current_price: float,
        timeframe: str,
    ) -> Optional[LiquidityZone]:
        upper = float(key_level.zone_boundaries[1])
        candidates = [sw for sw in swing_highs if float(sw.price) > upper]
        if not candidates:
            return None
        candidates.sort(key=lambda sw: abs(float(sw.price) - upper))
        max_distance = upper * 0.02
        for sw in candidates:
            distance = float(sw.price) - upper
            if distance > max_distance:
                continue
            if current_price > 0 and abs(float(sw.price) - current_price) / current_price > 0.15:
                continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="BSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"Retail stops above {timeframe} resistance (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                derived_from_sr_boundaries=(float(key_level.zone_boundaries[0]), float(key_level.zone_boundaries[1])),
                sr_timeframe=timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            )
        return None

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
        """
        SSL/BSL по SMC:
        - SSL = первый swing low ниже нижней границы support-зоны
        - BSL = первый swing high выше верхней границы resistance-зоны
        Приоритет: использовать zone_boundaries у S&R уровней и соответствующий таймфрейм.
        """
        liquidity_zones: List[LiquidityZone] = []
        current_price = self._get_current_price(structures)

        for sr in sr_levels:
            if sr.zone_boundaries is None:
                continue
            timeframe = sr.timeframe or "4H"
            if timeframe not in structures:
                continue
            struct = structures[timeframe]
            swing_highs = struct.all_swing_highs or []
            swing_lows = struct.all_swing_lows or []

            if sr.level_type == "support":
                zone = self._find_first_swing_below_support(sr, swing_lows, current_price)
                if zone:
                    liquidity_zones.append(zone)
            elif sr.level_type == "resistance":
                zone = self._find_first_swing_above_resistance(sr, swing_highs, current_price)
                if zone:
                    liquidity_zones.append(zone)

        return self._deduplicate_liquidity_zones(liquidity_zones)

    def _find_first_swing_below_support(
        self,
        sr_support: SupportResistanceLevel,
        swing_lows: List[SwingPoint],
        current_price: float,
    ) -> Optional[LiquidityZone]:
        if sr_support.zone_boundaries is None:
            return None
        lower = float(sr_support.zone_boundaries[0])
        candidates = [sw for sw in swing_lows if float(sw.price) < lower]
        if not candidates:
            return None
        candidates.sort(key=lambda sw: abs(float(sw.price) - lower))
        max_distance = lower * 0.02  # 2%
        for sw in candidates:
            distance = lower - float(sw.price)
            if distance > max_distance:
                continue
            if current_price > 0 and abs(float(sw.price) - current_price) / current_price > 0.15:
                continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="SSL",
                strength=float(sr_support.strength),
                estimated_volume="high" if (sr_support.obviousness_score or 0.0) > 0.8 else "medium",
                retail_logic=f"Retail stops below support at {sr_support.price:.6f} (zone: {sr_support.zone_boundaries[0]:.6f}-{sr_support.zone_boundaries[1]:.6f})",
                derived_from_sr_price=float(sr_support.price),
                derived_from_sr_boundaries=(float(sr_support.zone_boundaries[0]), float(sr_support.zone_boundaries[1])),
                sr_timeframe=sr_support.timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            )
        return None

    def _find_first_swing_above_resistance(
        self,
        sr_resistance: SupportResistanceLevel,
        swing_highs: List[SwingPoint],
        current_price: float,
    ) -> Optional[LiquidityZone]:
        if sr_resistance.zone_boundaries is None:
            return None
        upper = float(sr_resistance.zone_boundaries[1])
        candidates = [sw for sw in swing_highs if float(sw.price) > upper]
        if not candidates:
            return None
        candidates.sort(key=lambda sw: abs(float(sw.price) - upper))
        max_distance = upper * 0.02  # 2%
        for sw in candidates:
            distance = float(sw.price) - upper
            if distance > max_distance:
                continue
            if current_price > 0 and abs(float(sw.price) - current_price) / current_price > 0.15:
                continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="BSL",
                strength=float(sr_resistance.strength),
                estimated_volume="high" if (sr_resistance.obviousness_score or 0.0) > 0.8 else "medium",
                retail_logic=f"Retail stops above resistance at {sr_resistance.price:.6f} (zone: {sr_resistance.zone_boundaries[0]:.6f}-{sr_resistance.zone_boundaries[1]:.6f})",
                derived_from_sr_price=float(sr_resistance.price),
                derived_from_sr_boundaries=(float(sr_resistance.zone_boundaries[0]), float(sr_resistance.zone_boundaries[1])),
                sr_timeframe=sr_resistance.timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            )
        return None

    def _get_current_price(self, structures: Dict[str, StructureAnalysis]) -> float:
        for tf in ["4H", "1D"]:
            if tf in structures:
                s = structures[tf]
                try:
                    return float(max(float(s.last_swing_high.price), float(s.last_swing_low.price)))
                except Exception:
                    continue
        return 0.0

    def _deduplicate_liquidity_zones(self, zones: List[LiquidityZone]) -> List[LiquidityZone]:
        if not zones:
            return []
        zones_sorted = sorted(zones, key=lambda z: float(z.price))
        deduped: List[LiquidityZone] = [zones_sorted[0]]
        for z in zones_sorted[1:]:
            last = deduped[-1]
            if abs(float(z.price) - float(last.price)) / max(float(last.price), 1e-9) > 0.001:
                deduped.append(z)
        return deduped


