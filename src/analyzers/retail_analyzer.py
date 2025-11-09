from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from datetime import datetime

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
        self.logger = logging.getLogger(__name__)
        # Asset-specific defaults can be adjusted via _get_asset_params

    def analyze_retail_behavior(
        self,
        data_1d: pd.DataFrame,
        fibonacci: Dict[str, FibonacciAnalysis],
        structures: Dict[str, StructureAnalysis],
        data_h4: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
        data_h4_extended: Optional[pd.DataFrame] = None,
        data_1d_extended: Optional[pd.DataFrame] = None,
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
        runtime_current = float(current_price) if current_price is not None else self._get_current_price(structures)
        key_sr_levels = self.select_key_sr_levels(sr_levels_1d, sr_levels_4h, runtime_current)

        # 4) Retail entry анализ (можно использовать полный набор уровней)
        retail_entry_analysis = self.analyze_retail_entry_probability(
            fibonacci["1D"], sr_levels_1d + sr_levels_4h
        )

        # 5) SSL/BSL только от 4 ключевых уровней (с учётом текущей цены и сырых данных TF)
        liquidity_zones = self.identify_liquidity_zones_from_key_levels(
            structures,
            key_sr_levels,
            data_h4=data_h4,
            data_1d=data_1d,
            current_price=runtime_current,
            data_h4_full=data_h4_extended,
            data_1d_full=data_1d_extended,
        )

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
            lookback_candles=1000,
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
        data_h4: Optional[pd.DataFrame] = None,
        data_1d: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
        data_h4_full: Optional[pd.DataFrame] = None,
        data_1d_full: Optional[pd.DataFrame] = None,
    ) -> List[LiquidityZone]:
        liquidity_zones: List[LiquidityZone] = []
        self.logger.info("=== Identifying Liquidity Zones ===")
        runtime_current = float(current_price) if current_price is not None else self._get_current_price(structures)
        self.logger.info("Current price: %s", f"{runtime_current:.6f}")

        # helper to augment swings with raw pivots
        def _aug(swings: List[SwingPoint], df: Optional[pd.DataFrame], want_type: str, boundary: float, tf: str) -> List[SwingPoint]:
            base = list(swings or [])
            if df is None or len(df) < 5:
                return base
            # используем последние 300 свечей для более широкого анализа
            window = 1000 if tf == "1D" else 300
            df_use = df.tail(window) if len(df) > window else df
            extra = self._extract_local_extrema(df_use, want_type, boundary, pivot_span=2, tf=tf)
            # merge by price proximity 0.0005 (≈5e-4 ~ 5e-4 absolute)
            for sp in extra:
                if not any(abs(float(sp.price) - float(x.price)) / max(abs(float(sp.price)), 1e-9) < 5e-4 for x in base):
                    base.append(sp)
            return base

        # 1D support -> все SSL (потом ограничим количеством)
        if key_sr_levels.d1_support and "1D" in structures:
            struct = structures["1D"]
            lower = float(key_sr_levels.d1_support.zone_boundaries[0])
            upper = float(key_sr_levels.d1_support.zone_boundaries[1])
            src_1d = data_1d_full if data_1d_full is not None else data_1d
            # Диагностика источников данных
            try:
                self.logger.info("data_1d: %s", f"{len(data_1d)} candles" if data_1d is not None else "None")
                self.logger.info("data_1d_full: %s", f"{len(data_1d_full)} candles" if data_1d_full is not None else "None")
                if src_1d is not None and len(src_1d) > 0:
                    min_300 = float(src_1d.tail(300)["Low"].min()) if len(src_1d) >= 300 else float(src_1d["Low"].min())
                    min_1000 = float(src_1d.tail(1000)["Low"].min()) if len(src_1d) >= 1000 else min_300
                    min_all = float(src_1d["Low"].min())
                    self.logger.info("Min price: 300d=%.6f, 1000d=%.6f, all=%.6f", min_300, min_1000, min_all)
                    self.logger.info("Support lower boundary: %.6f | below: 300d=%s, 1000d=%s, all=%s",
                                     lower, str(min_300 < lower), str(min_1000 < lower), str(min_all < lower))
            except Exception:
                pass
            swing_lows = _aug(struct.all_swing_lows or [], src_1d, "low", lower, "1D")
            self.logger.info("1D Support zone: [%s, %s]", f"{lower:.6f}", f"{upper:.6f}")
            self.logger.info("Found %d 1D swing lows total", len(swing_lows))
            zones = self._find_all_ssl_from_key_level(
                key_sr_levels.d1_support,
                swing_lows,
                runtime_current,
                "1D",
                symbol=self.symbol,
                data=src_1d,
                sr_last_touch=key_sr_levels.d1_support.last_touch if key_sr_levels.d1_support else None,
            )
            # оставить первые по времени (макс 2)
            zones.sort(key=lambda z: getattr(z, "swing_timestamp", datetime.min))
            liquidity_zones.extend(zones[:2])
            # Fallback: если 1D SSL не найден — принудительно выбрать ближайший ниже зоны
            if not any(z.zone_type == "SSL" and z.timeframe == "1D" for z in liquidity_zones):
                below = [sp for sp in swing_lows if float(sp.price) < lower]
                self.logger.info("1D swing lows below zone: %d", len(below))
                if below:
                    for sp in below:
                        dist_pips = (lower - float(sp.price)) * 10000.0
                        self.logger.info("  - Swing Low: %.6f, distance from zone: %.1f pips", float(sp.price), dist_pips)
                    nearest = max(below, key=lambda sp: float(sp.price))
                    dist_curr = abs(float(nearest.price) - runtime_current) / max(runtime_current, 1e-9)
                    min_bound_ratio = 0.001  # 0.1% отступ от нижней границы
                    dist_bound_ratio = (lower - float(nearest.price)) / max(lower, 1e-9)
                    if dist_curr <= 0.20 and dist_bound_ratio >= min_bound_ratio:
                        self.logger.info("✅ 1D SSL added: %.6f (distance_from_current=%.2f%%)", float(nearest.price), dist_curr * 100.0)
                        liquidity_zones.append(
                            LiquidityZone(
                                price=float(nearest.price),
                                zone_type="SSL",
                                strength=float(key_sr_levels.d1_support.strength),
                                estimated_volume="high" if float(key_sr_levels.d1_support.obviousness_score) > 0.8 else "medium",
                                retail_logic=f"Retail stops below 1D support (zone: {lower:.6f}-{upper:.6f})",
                                timeframe="1D",
                                derived_from_sr_boundaries=(float(key_sr_levels.d1_support.zone_boundaries[0]), float(key_sr_levels.d1_support.zone_boundaries[1])),
                                sr_timeframe="1D",
                                swing_timestamp=getattr(nearest, "timestamp", None),
                                swing_strength=float(getattr(nearest, "strength", 0.0) or 0.0),
                            )
                        )
                    else:
                        self.logger.warning(
                            "❌ 1D SSL candidate rejected: dist_curr=%.2f%%, dist_bound=%.3f%%",
                            dist_curr * 100.0,
                            dist_bound_ratio * 100.0,
                        )
                else:
                    self.logger.warning("❌ No 1D swing lows found below support zone — trying full dataset")
                    # Попытка по всему доступному датасету (если _aug ограничил окно)
                    try:
                        if src_1d is not None and len(src_1d) > 0:
                            all_lows = self._extract_local_extrema(src_1d, "low", lower, pivot_span=2, tf="1D")
                            below_full = [sp for sp in all_lows if float(sp.price) < lower]
                            self.logger.info("Full dataset swing lows below zone: %d", len(below_full))
                            if below_full:
                                nearest = max(below_full, key=lambda sp: float(sp.price))
                                dist_curr = abs(float(nearest.price) - runtime_current) / max(runtime_current, 1e-9)
                                min_bound_ratio = 0.001
                                dist_bound_ratio = (lower - float(nearest.price)) / max(lower, 1e-9)
                                if dist_curr <= 0.20 and dist_bound_ratio >= min_bound_ratio:
                                    self.logger.info("✅ 1D SSL added (full): %.6f", float(nearest.price))
                                    liquidity_zones.append(
                                        LiquidityZone(
                                            price=float(nearest.price),
                                            zone_type="SSL",
                                            strength=float(key_sr_levels.d1_support.strength),
                                            estimated_volume="high" if float(key_sr_levels.d1_support.obviousness_score) > 0.8 else "medium",
                                            retail_logic=f"Retail stops below 1D support (zone: {lower:.6f}-{upper:.6f})",
                                            timeframe="1D",
                                            derived_from_sr_boundaries=(float(key_sr_levels.d1_support.zone_boundaries[0]), float(key_sr_levels.d1_support.zone_boundaries[1])),
                                            sr_timeframe="1D",
                                            swing_timestamp=getattr(nearest, "timestamp", None),
                                            swing_strength=float(getattr(nearest, "strength", 0.0) or 0.0),
                                        )
                                    )
                                else:
                                    self.logger.warning("❌ 1D SSL (full) rejected by distance constraints")
                            else:
                                self.logger.warning("❌ No 1D swing lows found below support zone in full dataset")
                    except Exception:
                        pass
                    # Если и после full ничего — расчётный SSL
                    if not any(z.zone_type == "SSL" and z.timeframe == "1D" for z in liquidity_zones):
                        ssl_price = lower * 0.997  # -0.3%
                        dist_curr = abs(ssl_price - runtime_current) / max(runtime_current, 1e-9)
                        dist_bound_ratio = (lower - ssl_price) / max(lower, 1e-9)
                        if dist_curr <= 0.20 and dist_bound_ratio >= 0.001:
                            self.logger.info("✅ 1D SSL added (calculated): %.6f", ssl_price)
                            liquidity_zones.append(
                                LiquidityZone(
                                    price=float(ssl_price),
                                    zone_type="SSL",
                                    strength=float(key_sr_levels.d1_support.strength),
                                    estimated_volume="high" if float(key_sr_levels.d1_support.obviousness_score) > 0.8 else "medium",
                                    retail_logic=f"Retail stops below 1D support (calculated from zone: {lower:.6f}-{upper:.6f})",
                                    timeframe="1D",
                                    derived_from_sr_boundaries=(float(key_sr_levels.d1_support.zone_boundaries[0]), float(key_sr_levels.d1_support.zone_boundaries[1])),
                                    sr_timeframe="1D",
                                    swing_timestamp=None,
                                    swing_strength=0.0,
                                )
                            )
                        else:
                            self.logger.warning("❌ Calculated 1D SSL rejected by distance constraints")
        # 1D resistance -> все BSL
        if key_sr_levels.d1_resistance and "1D" in structures:
            struct = structures["1D"]
            upper = float(key_sr_levels.d1_resistance.zone_boundaries[1])
            src_1d = data_1d_full if data_1d_full is not None else data_1d
            swing_highs = _aug(struct.all_swing_highs or [], src_1d, "high", upper, "1D")
            zones = self._find_all_bsl_from_key_level(
                key_sr_levels.d1_resistance,
                swing_highs,
                runtime_current,
                "1D",
                symbol=self.symbol,
                data=src_1d,
                sr_last_touch=key_sr_levels.d1_resistance.last_touch if key_sr_levels.d1_resistance else None,
            )
            zones.sort(key=lambda z: getattr(z, "swing_timestamp", datetime.min))
            liquidity_zones.extend(zones[:2])
        # 4H support -> все SSL (ограничить ближайшей 1)
        if key_sr_levels.h4_support and "4H" in structures:
            struct = structures["4H"]
            lower = float(key_sr_levels.h4_support.zone_boundaries[0])
            src_h4 = data_h4_full if data_h4_full is not None else data_h4
            swing_lows = _aug(struct.all_swing_lows or [], src_h4, "low", lower, "4H")
            zones = self._find_all_ssl_from_key_level(
                key_sr_levels.h4_support,
                swing_lows,
                runtime_current,
                "4H",
                symbol=self.symbol,
                data=src_h4,
                sr_last_touch=key_sr_levels.h4_support.last_touch if key_sr_levels.h4_support else None,
            )
            zones.sort(key=lambda z: getattr(z, "swing_timestamp", datetime.min))
            liquidity_zones.extend(zones[:1])
            # ATR-based fallback if none found for 4H SSL
            if not any(z.zone_type == "SSL" and z.timeframe == "4H" for z in liquidity_zones):
                self.logger.warning("❌ No 4H SSL found below support zone - using ATR/historical fallback")
                atr = self._calculate_atr(src_h4, period=14) if src_h4 is not None else 0.0
                params = self._get_asset_params(self.symbol, runtime_current)
                if atr > 0.0:
                    calculated_ssl = lower - (atr * params["atr_multiplier"])
                    self.logger.info("✅ 4H SSL added (ATR): %.6f (lower=%.6f - ATR=%.2f * %.1f)", calculated_ssl, lower, atr, params["atr_multiplier"])
                else:
                    historical_low = float(src_h4["Low"].min()) if src_h4 is not None and len(src_h4) > 0 else lower
                    if historical_low < lower:
                        calculated_ssl = historical_low
                        self.logger.info("✅ 4H SSL added (historical low): %.6f", calculated_ssl)
                    else:
                        calculated_ssl = lower * (1.0 - params["max_distance_ratio"] / 2.0)
                        self.logger.info("✅ 4H SSL added (percentage): %.6f", calculated_ssl)
                liquidity_zones.append(
                    LiquidityZone(
                        price=float(calculated_ssl),
                        zone_type="SSL",
                        strength=float(key_sr_levels.h4_support.strength),
                        estimated_volume="medium",
                        retail_logic=f"Retail stops below 4H support (zone: {lower:.6f}-{float(key_sr_levels.h4_support.zone_boundaries[1]):.6f}) [calculated]",
                        timeframe="4H",
                        derived_from_sr_boundaries=(
                            float(key_sr_levels.h4_support.zone_boundaries[0]),
                            float(key_sr_levels.h4_support.zone_boundaries[1]),
                        ),
                        sr_timeframe="4H",
                    )
                )
        # 4H resistance -> все BSL (ограничить ближайшей 1)
        if key_sr_levels.h4_resistance and "4H" in structures:
            struct = structures["4H"]
            upper = float(key_sr_levels.h4_resistance.zone_boundaries[1])
            src_h4 = data_h4_full if data_h4_full is not None else data_h4
            swing_highs = _aug(struct.all_swing_highs or [], src_h4, "high", upper, "4H")
            zones = self._find_all_bsl_from_key_level(
                key_sr_levels.h4_resistance,
                swing_highs,
                runtime_current,
                "4H",
                symbol=self.symbol,
                data=src_h4,
                sr_last_touch=key_sr_levels.h4_resistance.last_touch if key_sr_levels.h4_resistance else None,
            )
            zones.sort(key=lambda z: getattr(z, "swing_timestamp", datetime.min))
            liquidity_zones.extend(zones[:1])
            # ATR-based fallback if none found for 4H BSL
            if not any(z.zone_type == "BSL" and z.timeframe == "4H" for z in liquidity_zones):
                self.logger.warning("❌ No 4H BSL found above resistance zone - using ATR/historical fallback")
                atr = self._calculate_atr(src_h4, period=14) if src_h4 is not None else 0.0
                params = self._get_asset_params(self.symbol, runtime_current)
                if atr > 0.0:
                    calculated_bsl = upper + (atr * params["atr_multiplier"])
                    self.logger.info("✅ 4H BSL added (ATR): %.6f (upper=%.6f + ATR=%.2f * %.1f)", calculated_bsl, upper, atr, params["atr_multiplier"])
                else:
                    historical_high = float(src_h4["High"].max()) if src_h4 is not None and len(src_h4) > 0 else upper
                    if historical_high > upper:
                        calculated_bsl = historical_high
                        self.logger.info("✅ 4H BSL added (historical high): %.6f", calculated_bsl)
                    else:
                        calculated_bsl = upper * (1.0 + params["max_distance_ratio"] / 2.0)
                        self.logger.info("✅ 4H BSL added (percentage): %.6f", calculated_bsl)
                liquidity_zones.append(
                    LiquidityZone(
                        price=float(calculated_bsl),
                        zone_type="BSL",
                        strength=float(key_sr_levels.h4_resistance.strength),
                        estimated_volume="medium",
                        retail_logic=f"Retail stops above 4H resistance (zone: {float(key_sr_levels.h4_resistance.zone_boundaries[0]):.6f}-{upper:.6f}) [calculated]",
                        timeframe="4H",
                        derived_from_sr_boundaries=(
                            float(key_sr_levels.h4_resistance.zone_boundaries[0]),
                            float(key_sr_levels.h4_resistance.zone_boundaries[1]),
                        ),
                        sr_timeframe="4H",
                    )
                )

        return self._deduplicate_liquidity_zones(liquidity_zones)

    def _extract_local_extrema(
        self,
        df: pd.DataFrame,
        want_type: str,
        boundary: float,
        pivot_span: int,
        tf: str,
    ) -> List[SwingPoint]:
        res: List[SwingPoint] = []
        n = len(df)
        for i in range(pivot_span, n - pivot_span):
            if want_type == "low":
                val = float(df.iloc[i]["Low"])
                if not (val < float(boundary)):
                    continue
                left = float(df.iloc[i - pivot_span : i]["Low"].min())
                right = float(df.iloc[i + 1 : i + 1 + pivot_span]["Low"].min())
                if val <= left and val <= right:
                    res.append(SwingPoint(timestamp=df.index[i], price=val, type="low", timeframe=tf, strength=0.5))
            else:
                val = float(df.iloc[i]["High"])
                if not (val > float(boundary)):
                    continue
                left = float(df.iloc[i - pivot_span : i]["High"].max())
                right = float(df.iloc[i + 1 : i + 1 + pivot_span]["High"].max())
                if val >= left and val >= right:
                    res.append(SwingPoint(timestamp=df.index[i], price=val, type="high", timeframe=tf, strength=0.5))
        return res

    def _find_ssl_from_key_level(
        self,
        key_level: KeySRLevel,
        swing_lows: List[SwingPoint],
        current_price: float,
        timeframe: str,
        sr_last_touch: Optional[str] = None,
    ) -> Optional[LiquidityZone]:
        lower = float(key_level.zone_boundaries[0])
        candidates = [sw for sw in swing_lows if float(sw.price) < lower]
        if not candidates:
            return None
        # SSL должен быть ниже текущей цены
        if current_price > 0:
            candidates = [sw for sw in candidates if float(sw.price) < current_price]
            if not candidates:
                return None
        # Time-gate by last touch if present
        if sr_last_touch:
            try:
                cut = datetime.fromisoformat(sr_last_touch.replace("+00:00", ""))
                tfiltered = [sw for sw in candidates if getattr(sw, "timestamp", None) and sw.timestamp >= cut]
                if tfiltered:
                    candidates = tfiltered
            except Exception:
                pass
        # Sort chronologically (earliest first)
        candidates.sort(key=lambda sw: getattr(sw, "timestamp", datetime.min))
        max_distance = lower * (0.05 if timeframe == "1D" else 0.02)
        min_curr_ratio = 0.002 if timeframe == "1D" else 0.001  # 0.2% для 1D, 0.1% для 4H
        min_bound_ratio = 0.001 if timeframe == "1D" else 0.0005  # 0.1% для 1D, 0.05% для 4H
        for sw in candidates:
            distance = lower - float(sw.price)
            if distance > max_distance:
                continue
            if current_price > 0:
                dist_curr = abs(float(sw.price) - current_price) / current_price
                max_curr = 0.20 if timeframe == "1D" else 0.15
                # нижняя граница по расстоянию от текущей цены
                if dist_curr < min_curr_ratio or dist_curr > max_curr:
                    continue
            # нижняя граница по отступу от границы зоны
            if lower > 0:
                dist_bound = distance / lower
                if dist_bound < min_bound_ratio:
                    continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="SSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"First swing low below {timeframe} support (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                timeframe=timeframe,
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
        sr_last_touch: Optional[str] = None,
    ) -> Optional[LiquidityZone]:
        upper = float(key_level.zone_boundaries[1])
        candidates = [sw for sw in swing_highs if float(sw.price) > upper]
        if not candidates:
            return None
        # BSL должен быть выше текущей цены
        if current_price > 0:
            candidates = [sw for sw in candidates if float(sw.price) > current_price]
            if not candidates:
                return None
        # Time-gate by last touch if present
        if sr_last_touch:
            try:
                cut = datetime.fromisoformat(sr_last_touch.replace("+00:00", ""))
                tfiltered = [sw for sw in candidates if getattr(sw, "timestamp", None) and sw.timestamp >= cut]
                if tfiltered:
                    candidates = tfiltered
            except Exception:
                pass
        # Sort chronologically (earliest first)
        candidates.sort(key=lambda sw: getattr(sw, "timestamp", datetime.min))
        max_distance = upper * (0.05 if timeframe == "1D" else 0.02)
        min_curr_ratio = 0.002 if timeframe == "1D" else 0.001
        min_bound_ratio = 0.001 if timeframe == "1D" else 0.0005
        for sw in candidates:
            distance = float(sw.price) - upper
            if distance > max_distance:
                continue
            if current_price > 0:
                dist_curr = abs(float(sw.price) - current_price) / current_price
                max_curr = 0.20 if timeframe == "1D" else 0.15
                if dist_curr < min_curr_ratio or dist_curr > max_curr:
                    continue
            if upper > 0:
                dist_bound = distance / upper
                if dist_bound < min_bound_ratio:
                    continue
            return LiquidityZone(
                price=float(sw.price),
                zone_type="BSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"First swing high above {timeframe} resistance (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                timeframe=timeframe,
                derived_from_sr_boundaries=(float(key_level.zone_boundaries[0]), float(key_level.zone_boundaries[1])),
                sr_timeframe=timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            )
        return None

    def _find_all_ssl_from_key_level(
        self,
        key_level: KeySRLevel,
        swing_lows: List[SwingPoint],
        current_price: float,
        timeframe: str,
        symbol: str = "UNKNOWN",
        data: Optional[pd.DataFrame] = None,
        sr_last_touch: Optional[str] = None,
    ) -> List[LiquidityZone]:
        lower = float(key_level.zone_boundaries[0])
        candidates = [sw for sw in swing_lows if float(sw.price) < lower]
        if not candidates:
            return []
        if current_price > 0:
            candidates = [sw for sw in candidates if float(sw.price) < current_price]
            if not candidates:
                return []
        # Time-gate by last touch if present
        if sr_last_touch:
            try:
                cut = datetime.fromisoformat(sr_last_touch.replace("+00:00", ""))
                tfiltered = [sw for sw in candidates if getattr(sw, "timestamp", None) and sw.timestamp >= cut]
                if tfiltered:
                    candidates = tfiltered
            except Exception:
                pass
        # Sort chronologically (earliest first)
        candidates.sort(key=lambda sw: getattr(sw, "timestamp", datetime.min))
        # Asset-specific parameters and ATR-based distance
        params = self._get_asset_params(symbol, current_price)
        atr = self._calculate_atr(data, period=14) if data is not None else 0.0
        max_distance = (atr * params["atr_multiplier"]) if atr and atr > 0.0 else (lower * params["max_distance_ratio"])
        min_curr_ratio = 0.002 if timeframe == "1D" else 0.001
        min_bound_ratio = params["min_bound_ratio"]
        zones: List[LiquidityZone] = []
        for sw in candidates:
            distance = lower - float(sw.price)
            if distance > max_distance:
                continue
            if current_price > 0:
                dist_curr = abs(float(sw.price) - current_price) / current_price
                max_curr = 0.20 if timeframe == "1D" else 0.15
                if dist_curr < min_curr_ratio or dist_curr > max_curr:
                    continue
            if lower > 0:
                dist_bound = distance / lower
                if dist_bound < min_bound_ratio:
                    continue
            zones.append(LiquidityZone(
                price=float(sw.price),
                zone_type="SSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"Retail stops below {timeframe} support (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                timeframe=timeframe,
                derived_from_sr_boundaries=(float(key_level.zone_boundaries[0]), float(key_level.zone_boundaries[1])),
                sr_timeframe=timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            ))
        return zones

    def _find_all_bsl_from_key_level(
        self,
        key_level: KeySRLevel,
        swing_highs: List[SwingPoint],
        current_price: float,
        timeframe: str,
        symbol: str = "UNKNOWN",
        data: Optional[pd.DataFrame] = None,
        sr_last_touch: Optional[str] = None,
    ) -> List[LiquidityZone]:
        upper = float(key_level.zone_boundaries[1])
        candidates = [sw for sw in swing_highs if float(sw.price) > upper]
        if not candidates:
            return []
        if current_price > 0:
            candidates = [sw for sw in candidates if float(sw.price) > current_price]
            if not candidates:
                return []
        # Time-gate by last touch if present
        if sr_last_touch:
            try:
                cut = datetime.fromisoformat(sr_last_touch.replace("+00:00", ""))
                tfiltered = [sw for sw in candidates if getattr(sw, "timestamp", None) and sw.timestamp >= cut]
                if tfiltered:
                    candidates = tfiltered
            except Exception:
                pass
        # Sort chronologically (earliest first)
        candidates.sort(key=lambda sw: getattr(sw, "timestamp", datetime.min))
        # Asset-specific parameters and ATR-based distance
        params = self._get_asset_params(symbol, current_price)
        atr = self._calculate_atr(data, period=14) if data is not None else 0.0
        max_distance = (atr * params["atr_multiplier"]) if atr and atr > 0.0 else (upper * params["max_distance_ratio"])
        min_curr_ratio = 0.002 if timeframe == "1D" else 0.001
        min_bound_ratio = params["min_bound_ratio"]
        zones: List[LiquidityZone] = []
        for sw in candidates:
            distance = float(sw.price) - upper
            if distance > max_distance:
                continue
            if current_price > 0:
                dist_curr = abs(float(sw.price) - current_price) / current_price
                max_curr = 0.20 if timeframe == "1D" else 0.15
                if dist_curr < min_curr_ratio or dist_curr > max_curr:
                    continue
            if upper > 0:
                dist_bound = distance / upper
                if dist_bound < min_bound_ratio:
                    continue
            zones.append(LiquidityZone(
                price=float(sw.price),
                zone_type="BSL",
                strength=float(key_level.strength),
                estimated_volume="high" if float(key_level.obviousness_score) > 0.8 else "medium",
                retail_logic=f"Retail stops above {timeframe} resistance (zone: {key_level.zone_boundaries[0]:.6f}-{key_level.zone_boundaries[1]:.6f})",
                timeframe=timeframe,
                derived_from_sr_boundaries=(float(key_level.zone_boundaries[0]), float(key_level.zone_boundaries[1])),
                sr_timeframe=timeframe,
                swing_timestamp=getattr(sw, "timestamp", None),
                swing_strength=float(getattr(sw, "strength", 0.0) or 0.0),
            ))
        return zones

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
        # Группируем по (type, timeframe)
        groups: Dict[tuple, List[LiquidityZone]] = {}
        for z in zones:
            key = (getattr(z, "zone_type", None), getattr(z, "timeframe", None))
            groups.setdefault(key, []).append(z)
        deduplicated: List[LiquidityZone] = []
        for (zone_type, timeframe), group in groups.items():
            if not group:
                continue
            group.sort(key=lambda z: float(z.price))
            filtered: List[LiquidityZone] = []
            for z in group:
                if not filtered:
                    filtered.append(z)
                    continue
                prev = filtered[-1]
                p_curr = float(z.price)
                p_prev = float(prev.price)
                # Порог: форекс — 50 пипсов, иначе 0.1% от цены
                if max(p_curr, p_prev) < 10.0:
                    dist_pips = abs(p_curr - p_prev) * 10000.0
                    too_close = dist_pips < 50.0
                else:
                    too_close = abs(p_curr - p_prev) / max(p_prev, 1e-9) < 0.001
                if not too_close:
                    filtered.append(z)
                else:
                    # Оставить ближайший к своей исходной зоне (если есть boundaries), иначе по силе
                    def _dist_to_boundary(q: LiquidityZone) -> float:
                        b = getattr(q, "derived_from_sr_boundaries", None)
                        if not b:
                            return float("inf")
                        low, high = float(b[0]), float(b[1])
                        # Для SSL ближе к low, для BSL ближе к high
                        if getattr(q, "zone_type", "") == "SSL":
                            return abs(float(q.price) - low)
                        return abs(float(q.price) - high)
                    d_prev = _dist_to_boundary(prev)
                    d_curr = _dist_to_boundary(z)
                    if d_curr < d_prev:
                        filtered[-1] = z
                    elif d_curr == float("inf") and d_prev == float("inf"):
                        # fallback: по strength
                        if float(getattr(z, "strength", 0.0) or 0.0) > float(getattr(prev, "strength", 0.0) or 0.0):
                            filtered[-1] = z
            deduplicated.extend(filtered)
        # Сортировка обратно по цене
        deduplicated.sort(key=lambda z: float(z.price))
        # Гарантировать максимум 1 1D SSL — оставим ближайший к границе SR
        try:
            one_d_ssl = [z for z in deduplicated if getattr(z, "zone_type", "") == "SSL" and getattr(z, "timeframe", "") == "1D"]
            if len(one_d_ssl) > 1:
                def _dist_to_low_boundary(q: LiquidityZone) -> float:
                    b = getattr(q, "derived_from_sr_boundaries", None)
                    if not b:
                        return float("inf")
                    return abs(float(q.price) - float(b[0]))
                keep = min(one_d_ssl, key=_dist_to_low_boundary)
                deduplicated = [z for z in deduplicated if not (getattr(z, "zone_type", "") == "SSL" and getattr(z, "timeframe", "") == "1D")]
                deduplicated.append(keep)
                deduplicated.sort(key=lambda z: float(z.price))
        except Exception:
            pass
        return deduplicated

    # ===== ASSET-SPECIFIC HELPERS =====
    def _get_asset_params(self, symbol: str, price: float) -> Dict[str, float]:
        """Return asset-specific thresholds for SSL/BSL detection."""
        s = (symbol or "").upper()
        # Forex
        if any(x in s for x in ["USD", "=X", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]):
            return {
                "min_bound_ratio": 0.0005,   # 0.05%
                "max_distance_ratio": 0.02,  # 2%
                "atr_multiplier": 1.5,
            }
        # Index futures
        if any(x in s for x in ["MNQ", "NQ", "ES", "YM", "RTY", "NASDAQ", "SPX", "^GSPC"]):
            return {
                "min_bound_ratio": 0.005,    # 0.5%
                "max_distance_ratio": 0.05,  # 5%
                "atr_multiplier": 2.5,
            }
        # Commodities
        if any(x in s for x in ["GC", "XAUUSD", "GOLD", "SI", "SILVER", "CL", "OIL"]):
            return {
                "min_bound_ratio": 0.003,    # 0.3%
                "max_distance_ratio": 0.03,  # 3%
                "atr_multiplier": 2.0,
            }
        # Crypto
        if any(x in s for x in ["BTC", "ETH", "USDT", "CRYPTO"]):
            return {
                "min_bound_ratio": 0.01,     # 1%
                "max_distance_ratio": 0.10,  # 10%
                "atr_multiplier": 3.0,
            }
        # Default
        return {
            "min_bound_ratio": 0.001,       # 0.1%
            "max_distance_ratio": 0.02,     # 2%
            "atr_multiplier": 2.0,
        }

    def _calculate_atr(self, data: Optional[pd.DataFrame], period: int = 14) -> float:
        """Compute ATR(period) on given OHLC data for dynamic thresholds."""
        try:
            if data is None or len(data) < 2:
                return 0.0
            high = data["High"].astype(float)
            low = data["Low"].astype(float)
            close_prev = data["Close"].astype(float).shift(1)
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean().iloc[-1]
            return float(0.0 if pd.isna(atr) else atr)
        except Exception:
            return 0.0


