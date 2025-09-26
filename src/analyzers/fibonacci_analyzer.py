from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..core.data_structures import FibonacciAnalysis, FibonacciZone, StructureAnalysis


class MultiTimeframeFibonacciAnalyzer:
    def __init__(self):
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.ote_range: Tuple[float, float] = (0.618, 0.786)

    def analyze_all_timeframes(
        self, structures: Dict[str, StructureAnalysis], current_price: float
    ) -> Dict[str, FibonacciAnalysis]:
        """
        ГЛАВНАЯ ФУНКЦИЯ: Fibonacci анализ для всех таймфреймов
        """
        results: Dict[str, FibonacciAnalysis] = {}

        for timeframe, structure in structures.items():
            fib_analysis = self.calculate_fibonacci_retracement(
                structure.last_swing_high.price,
                structure.last_swing_low.price,
                current_price,
                timeframe,
            )
            results[timeframe] = fib_analysis

        return results

    def calculate_fibonacci_retracement(
        self, swing_high: float, swing_low: float, current_price: float, timeframe: str
    ) -> FibonacciAnalysis:
        """
        Расчет Fibonacci уровней и определение текущей зоны
        """
        if swing_high == swing_low:
            raise ValueError("Swing high and low cannot be equal")

        # Определяем направление свинга явно (bullish: low->high, bearish: high->low)
        is_bullish = swing_high > swing_low
        if is_bullish:  # bullish leg
            range_low = swing_low
            range_high = swing_high
            range_size = range_high - range_low
            retracement = (current_price - range_low) / range_size
        else:  # bearish leg
            range_low = swing_high
            range_high = swing_low
            range_size = range_high - range_low
            retracement = (range_high - current_price) / range_size

        levels: Dict[str, float] = {}
        for level in self.fib_levels:
            levels[str(level)] = range_low + (range_size * level)

        current_zone = self._determine_fibonacci_zone(retracement, is_bullish)

        ote_consolidation = False
        if timeframe == "4H" and current_zone == FibonacciZone.OTE:
            # Заглушка: детекция консолидации будет реализована на данных H4 в анализаторе манипуляций
            ote_consolidation = False

        return FibonacciAnalysis(
            timeframe=timeframe,
            swing_high=float(swing_high),
            swing_low=float(swing_low),
            current_zone=current_zone,
            retracement_level=float(retracement),
            key_levels=levels,
            ote_consolidation=ote_consolidation,
        )

    def _determine_fibonacci_zone(self, retracement: float, is_bullish: bool) -> FibonacciZone:
        """Определение текущей Fibonacci зоны"""
        if retracement > 1.0:
            return FibonacciZone.EXTENSION_ABOVE if is_bullish else FibonacciZone.EXTENSION_BELOW
        if retracement < 0.0:
            return FibonacciZone.EXTENSION_BELOW if is_bullish else FibonacciZone.EXTENSION_ABOVE
        if self.ote_range[0] <= retracement <= self.ote_range[1]:
            return FibonacciZone.OTE
        elif retracement > 0.5:
            return FibonacciZone.PREMIUM
        elif retracement < 0.5:
            return FibonacciZone.DISCOUNT
        else:
            return FibonacciZone.EQUILIBRIUM

    def detect_ote_consolidation(self, data: pd.DataFrame, ote_levels: Tuple[float, float]) -> bool:
        """
        Определение консолидации в OTE зоне

        Критерии:
        - Цена находится в OTE диапазоне несколько периодов
        - Низкая волатильность
        - Горизонтальное движение
        """
        if data is None or len(data) == 0:
            return False

        recent_data = data.tail(20)
        if len(recent_data) == 0:
            return False

        in_ote_count = int(
            ((recent_data["Close"] >= ote_levels[0]) & (recent_data["Close"] <= ote_levels[1])).sum()
        )

        if in_ote_count / len(recent_data) > 0.7:
            price_range = float(recent_data["High"].max() - recent_data["Low"].min())
            ote_range_size = float(ote_levels[1] - ote_levels[0])

            if price_range <= ote_range_size * 1.2:
                return True

        return False


