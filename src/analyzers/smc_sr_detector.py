from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..core.data_structures import SupportResistanceLevel


@dataclass
class _Zone:
    price: float
    low: float
    high: float
    level_type: str  # "support" | "resistance"


class SmartMoneySRDetector:
    def __init__(
        self,
        lookback_candles: int = 300,
        min_touches: int = 3,
        max_touches: int = 10,
        tolerance_percent: float = 0.001,  # ~0.1%
        min_reaction_percent: float = 0.002,  # ~0.2%
        min_time_separation_hours: float = 12.0,
        max_levels: int = 6,
        pivot_span: int = 2,
    ) -> None:
        self.lookback_candles = lookback_candles
        self.min_touches = min_touches
        self.max_touches = max_touches
        self.tolerance_percent = tolerance_percent
        self.min_reaction_percent = min_reaction_percent
        self.min_time_separation_hours = min_time_separation_hours
        self.max_levels = max_levels
        self.pivot_span = pivot_span
        # SMC filters
        self.max_distance_percent: float = 0.15
        self.min_obviousness: float = 0.6

    def detect(self, data: pd.DataFrame, timeframe: str = "4H") -> List[SupportResistanceLevel]:
        if data is None or len(data) < max(50, self.min_touches * 10):
            return []

        df = data.tail(self.lookback_candles).copy()
        df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()
        if df.empty:
            return []

        current_price = float(df["Close"].iloc[-1])
        tol_abs = current_price * self.tolerance_percent
        react_abs = current_price * self.min_reaction_percent

        support_points: List[Tuple[float, pd.Timestamp]] = []
        resistance_points: List[Tuple[float, pd.Timestamp]] = []
        span = self.pivot_span
        for i in range(span, len(df) - span):
            low_i = float(df["Low"].iloc[i])
            if low_i == low_i and low_i < float(df["Low"].iloc[i - span : i].min()) and low_i < float(
                df["Low"].iloc[i + 1 : i + 1 + span].min()
            ):
                support_points.append((low_i, df.index[i]))
            high_i = float(df["High"].iloc[i])
            if high_i == high_i and high_i > float(df["High"].iloc[i - span : i].max()) and high_i > float(
                df["High"].iloc[i + 1 : i + 1 + span].max()
            ):
                resistance_points.append((high_i, df.index[i]))

        support_zones = self._cluster_points(support_points, tol_abs, level_type="support")
        resistance_zones = self._cluster_points(resistance_points, tol_abs, level_type="resistance")

        levels: List[SupportResistanceLevel] = []
        for zone in support_zones + resistance_zones:
            touches, ts_list, reactions, sep_hours = self._count_valid_touches(df, zone, react_abs)
            if touches < self.min_touches:
                continue
            if touches > self.max_touches:
                # Сжать консолидацию: оставим первые self.max_touches касаний по времени
                keep = self.max_touches
                ts_list = ts_list[:keep]
                reactions = reactions[:keep]
                sep_hours = sep_hours[: max(0, keep - 1)]
                touches = keep

            strength = min(1.0, (touches / float(self.max_touches)) * 0.6 + self._recent_weight(ts_list) * 0.4)
            obviousness = min(1.0, 0.3 * (touches / float(self.max_touches)) + 0.3 * self._avg_positive(reactions, default=0.0) / max(react_abs, 1e-9) + 0.4 * self._recent_weight(ts_list))
            last_touch_iso = ts_list[-1].isoformat() if ts_list else None
            distance_percent = abs(zone.price - current_price) / max(current_price, 1e-9)

            levels.append(
                SupportResistanceLevel(
                    price=float(zone.price),
                    touches=int(touches),
                    strength=float(strength),
                    level_type=zone.level_type,
                    retail_likely_to_trade=(zone.level_type == "support"),
                    timeframe=timeframe,
                    zone_boundaries=(float(zone.low), float(zone.high)),
                    obviousness_score=float(obviousness),
                    touch_timestamps=[t.isoformat() for t in ts_list],
                    last_touch=last_touch_iso,
                    reaction_strengths=[float(x) for x in reactions],
                    time_separation_hours=[float(x) for x in sep_hours],
                    distance_percent=float(distance_percent),
                )
            )

        # Фильтрация по расстоянию от цены и минимальной очевидности
        filtered = [
            lvl
            for lvl in levels
            if (lvl.distance_percent is None or lvl.distance_percent <= self.max_distance_percent)
            and (lvl.obviousness_score is None or lvl.obviousness_score >= self.min_obviousness)
        ]
        # Отсортировать: по strength, затем по близости к цене, ограничить max_levels
        filtered.sort(key=lambda x: (x.strength, -abs(x.price - current_price)), reverse=True)
        return filtered[: self.max_levels]

    def _cluster_points(self, points: List[Tuple[float, pd.Timestamp]], tol_abs: float, level_type: str) -> List[_Zone]:
        if not points:
            return []
        points_sorted = sorted(points, key=lambda x: x[0])
        groups: List[List[Tuple[float, pd.Timestamp]]] = [[points_sorted[0]]]
        for p, ts in points_sorted[1:]:
            if abs(p - groups[-1][-1][0]) <= tol_abs:
                groups[-1].append((p, ts))
            else:
                groups.append([(p, ts)])
        zones: List[_Zone] = []
        for grp in groups:
            prices = [p for p, _ in grp]
            zones.append(
                _Zone(
                    price=float(np.mean(prices)),
                    low=float(min(prices)),
                    high=float(max(prices)),
                    level_type=level_type,
                )
            )
        return zones

    def _count_valid_touches(
        self, df: pd.DataFrame, zone: _Zone, react_abs: float
    ) -> Tuple[int, List[pd.Timestamp], List[float], List[float]]:
        touches = 0
        timestamps: List[pd.Timestamp] = []
        reactions: List[float] = []
        sep_hours: List[float] = []
        last_ts: pd.Timestamp | None = None
        for i in range(1, len(df) - 1):
            hi = float(df["High"].iloc[i])
            lo = float(df["Low"].iloc[i])
            prev_close = float(df["Close"].iloc[i - 1])
            ts = df.index[i]
            if zone.level_type == "support":
                in_zone = (zone.low <= lo <= zone.high) or (zone.low <= float(df["Close"].iloc[i]) <= zone.high)
                approached_outside = prev_close > zone.high
                if in_zone and approached_outside:
                    react = self._measure_reaction(df, i, upwards=True)
                    if react >= react_abs and self._enough_time_separated(last_ts, ts):
                        touches += 1
                        timestamps.append(ts)
                        reactions.append(react)
                        if last_ts is not None:
                            sep_hours.append((ts - last_ts).total_seconds() / 3600.0)
                        last_ts = ts
            else:
                in_zone = (zone.low <= hi <= zone.high) or (zone.low <= float(df["Close"].iloc[i]) <= zone.high)
                approached_outside = prev_close < zone.low
                if in_zone and approached_outside:
                    react = self._measure_reaction(df, i, upwards=False)
                    if react >= react_abs and self._enough_time_separated(last_ts, ts):
                        touches += 1
                        timestamps.append(ts)
                        reactions.append(react)
                        if last_ts is not None:
                            sep_hours.append((ts - last_ts).total_seconds() / 3600.0)
                        last_ts = ts
        return touches, timestamps, reactions, sep_hours

    def _measure_reaction(self, df: pd.DataFrame, i: int, upwards: bool) -> float:
        curr_price = float(df["Close"].iloc[i])
        # Оценим реакции в пределах ближайших 3 свечей
        window = df.iloc[i + 1 : min(len(df), i + 4)]
        if window.empty:
            return 0.0
        if upwards:
            target = float(window["High"].max())
            return max(0.0, target - curr_price)
        else:
            target = float(window["Low"].min())
            return max(0.0, curr_price - target)

    def _enough_time_separated(self, last_ts: pd.Timestamp | None, ts: pd.Timestamp) -> bool:
        if last_ts is None:
            return True
        return (ts - last_ts).total_seconds() / 3600.0 >= self.min_time_separation_hours

    def _recent_weight(self, ts_list: List[pd.Timestamp]) -> float:
        if not ts_list:
            return 0.0
        # Чем свежее последнее касание, тем выше вес (нормируем на 30 дней)
        now = ts_list[-1]
        first = ts_list[0]
        span_hours = max(1.0, (now - first).total_seconds() / 3600.0)
        # Чем меньше интервал между первым и последним — тем выше «концентрация» (в пределах разумного)
        score = min(1.0, 240.0 / span_hours)
        return score

    def _avg_positive(self, arr: List[float], default: float = 0.0) -> float:
        if not arr:
            return default
        vals = [x for x in arr if x > 0]
        return float(np.mean(vals)) if vals else default


