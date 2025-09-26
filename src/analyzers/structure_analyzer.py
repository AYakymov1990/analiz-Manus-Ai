import numpy as np
import pandas as pd
from typing import List, Dict

from ..core.data_structures import (
    SwingPoint,
    StructureAnalysis,
    StructureDirection,
)


class MultiTimeframeStructureAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Адаптивные параметры для разных таймфреймов
        self.swing_windows = {
            "1D": 5,
            "4H": 3,
            "15M": 2,
        }

    def analyze_all_timeframes(
        self,
        data_1d: pd.DataFrame,
        data_h4: pd.DataFrame,
        data_m15: pd.DataFrame,
    ) -> Dict[str, StructureAnalysis]:
        """
        ГЛАВНАЯ ФУНКЦИЯ: Анализ структуры на всех таймфреймах

        Возвращает словарь с анализом для каждого таймфрейма
        """
        results: Dict[str, StructureAnalysis] = {}

        swings_1d = self.detect_swing_points(data_1d, "1D")
        results["1D"] = self.determine_structure(swings_1d, "1D")

        swings_h4 = self.detect_swing_points(data_h4, "4H")
        results["4H"] = self.determine_structure(swings_h4, "4H")

        swings_m15 = self.detect_swing_points(data_m15, "15M")
        results["15M"] = self.determine_structure(swings_m15, "15M")

        return results

    def detect_swing_points(self, data: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
        """
        Определение swing high/low точек

        Алгоритм:
        1. Использовать rolling window для поиска локальных экстремумов
        2. Фильтровать слабые swing точки
        3. Рассчитать силу каждой swing точки
        """
        if data is None or len(data) == 0:
            return []

        # Требуемые колонки
        required_columns = ["High", "Low", "Close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        swing_points: List[SwingPoint] = []
        window = self.swing_windows[timeframe]

        # Поиск swing highs
        for i in range(window, len(data) - window):
            if self._is_swing_high(data, i, window):
                strength = self._calculate_swing_strength(data, i, "high", window)
                swing_points.append(
                    SwingPoint(
                        timestamp=data.index[i],
                        price=float(data.iloc[i]["High"]),
                        type="high",
                        timeframe=timeframe,
                        strength=float(strength),
                    )
                )

        # Поиск swing lows
        for i in range(window, len(data) - window):
            if self._is_swing_low(data, i, window):
                strength = self._calculate_swing_strength(data, i, "low", window)
                swing_points.append(
                    SwingPoint(
                        timestamp=data.index[i],
                        price=float(data.iloc[i]["Low"]),
                        type="low",
                        timeframe=timeframe,
                        strength=float(strength),
                    )
                )

        swing_points.sort(key=lambda x: x.timestamp)
        return swing_points

    def _is_swing_high(self, data: pd.DataFrame, index: int, window: int) -> bool:
        """Проверка является ли точка swing high"""
        current_high = float(data.iloc[index]["High"])
        # Узкое соседнее окно для локального экстремума
        k = 1
        left_n = data.iloc[index - k : index]
        right_n = data.iloc[index + 1 : index + k + 1]
        if len(left_n) < k or len(right_n) < k:
            return False
        left_max_n = float(left_n["High"].max())
        right_max_n = float(right_n["High"].max())
        eps = 1e-12
        # Разрешаем плато: текущее значение не меньше соседних максимумов
        cond_local = (current_high >= left_max_n - eps) and (current_high >= right_max_n - eps)
        if not cond_local:
            return False
        return True

    def _is_swing_low(self, data: pd.DataFrame, index: int, window: int) -> bool:
        """Проверка является ли точка swing low"""
        current_low = float(data.iloc[index]["Low"])
        k = 1
        left_n = data.iloc[index - k : index]
        right_n = data.iloc[index + 1 : index + k + 1]
        if len(left_n) < k or len(right_n) < k:
            return False
        left_min_n = float(left_n["Low"].min())
        right_min_n = float(right_n["Low"].min())
        eps = 1e-12
        # Разрешаем плато: текущее значение не больше соседних минимумов
        cond_local = (current_low <= left_min_n + eps) and (current_low <= right_min_n + eps)
        if not cond_local:
            return False
        return True

    def _calculate_swing_strength(
        self, data: pd.DataFrame, index: int, swing_type: str, window: int
    ) -> float:
        """
        Расчет силы swing точки (0-1)

        Факторы:
        - Размер движения до swing точки
        - Объем (если доступен)
        - Время формирования
        """
        if swing_type == "high":
            price_move = float(data.iloc[index]["High"]) - float(data.iloc[index - window]["Low"])
        else:
            price_move = float(data.iloc[index - window]["High"]) - float(data.iloc[index]["Low"])

        atr = float(self._calculate_atr(data, 14))
        if np.isnan(atr) or atr <= 0:
            # Фолбэк: используем средний диапазон свечи
            tr = (data["High"] - data["Low"]).rolling(window=window).mean().iloc[index]
            atr = float(tr) if not np.isnan(tr) and tr > 0 else 1e-9

        strength = max(0.0, min(price_move / (atr * 2.0), 1.0))
        return float(strength)

    def determine_structure(self, swing_points: List[SwingPoint], timeframe: str) -> StructureAnalysis:
        """
        Определение направления структуры рынка

        Логика:
        - Bullish: Higher Highs + Higher Lows
        - Bearish: Lower Highs + Lower Lows
        - Sideways: смешанные сигналы
        """
        if len(swing_points) < 4:
            return self._create_undefined_structure(timeframe)

        recent_swings = swing_points[-6:]
        highs = [s for s in recent_swings if s.type == "high"]
        lows = [s for s in recent_swings if s.type == "low"]

        if len(highs) < 2 or len(lows) < 2:
            return self._create_undefined_structure(timeframe)

        higher_highs = self._check_higher_highs(highs)
        higher_lows = self._check_higher_lows(lows)
        lower_highs = self._check_lower_highs(highs)
        lower_lows = self._check_lower_lows(lows)

        if higher_highs and higher_lows:
            direction = StructureDirection.BULLISH
            confidence = 0.8
        elif lower_highs and lower_lows:
            direction = StructureDirection.BEARISH
            confidence = 0.8
        else:
            direction = StructureDirection.SIDEWAYS
            confidence = 0.5

        if direction == StructureDirection.BULLISH:
            break_level = min([s.price for s in lows[-2:]])
        elif direction == StructureDirection.BEARISH:
            break_level = max([s.price for s in highs[-2:]])
        else:
            break_level = (highs[-1].price + lows[-1].price) / 2.0

        return StructureAnalysis(
            timeframe=timeframe,
            direction=direction,
            last_swing_high=highs[-1],
            last_swing_low=lows[-1],
            structure_strength=self._calculate_structure_strength(recent_swings),
            break_level=float(break_level),
            confidence=float(confidence),
        )

    def _check_higher_highs(self, highs: List[SwingPoint]) -> bool:
        if len(highs) < 2:
            return False
        return highs[-1].price > highs[-2].price

    def _check_higher_lows(self, lows: List[SwingPoint]) -> bool:
        if len(lows) < 2:
            return False
        return lows[-1].price > lows[-2].price

    def _check_lower_highs(self, highs: List[SwingPoint]) -> bool:
        if len(highs) < 2:
            return False
        return highs[-1].price < highs[-2].price

    def _check_lower_lows(self, lows: List[SwingPoint]) -> bool:
        if len(lows) < 2:
            return False
        return lows[-1].price < lows[-2].price

    def _calculate_structure_strength(self, swings: List[SwingPoint]) -> float:
        # Простейшая метрика: средняя "strength" последних swing
        if not swings:
            return 0.0
        strengths = [max(0.0, min(1.0, s.strength)) for s in swings]
        return float(np.mean(strengths))

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Расчет Average True Range"""
        high = data["High"].astype(float)
        low = data["Low"].astype(float)
        close = data["Close"].astype(float)

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_series = pd.Series(true_range).rolling(window=period, min_periods=1).mean()
        return float(atr_series.iloc[-1])

    def _create_undefined_structure(self, timeframe: str) -> StructureAnalysis:
        # Заглушки для неопределенной структуры: используем нулевые значения
        dummy_time = pd.Timestamp("1970-01-01")
        dummy_high = SwingPoint(timestamp=dummy_time, price=0.0, type="high", timeframe=timeframe, strength=0.0)
        dummy_low = SwingPoint(timestamp=dummy_time, price=0.0, type="low", timeframe=timeframe, strength=0.0)
        return StructureAnalysis(
            timeframe=timeframe,
            direction=StructureDirection.SIDEWAYS,
            last_swing_high=dummy_high,
            last_swing_low=dummy_low,
            structure_strength=0.0,
            break_level=0.0,
            confidence=0.0,
        )


