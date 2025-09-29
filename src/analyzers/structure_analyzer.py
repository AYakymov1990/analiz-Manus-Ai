import numpy as np
import pandas as pd
import logging
from typing import List, Dict

from ..core.data_structures import (
    SwingPoint,
    StructureAnalysis,
    StructureDirection,
)


class MultiTimeframeStructureAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
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

        for tf, df in (("1D", data_1d), ("4H", data_h4), ("15M", data_m15)):
            try:
                swings = self.detect_swing_points(df, tf)
                # fallback, если свингов мало
                if len(swings) < 2:
                    self.logger.warning("Недостаточно swing точек для %s, используем window extremes fallback", tf)
                    swings = self._get_window_extremes_as_swings(df, tf)
                analysis = self.determine_structure(swings, tf)
                results[tf] = self._validate_and_correct(df, analysis)
            except Exception as ex:
                self.logger.error("Ошибка анализа %s: %s. Применяем критический fallback.", tf, ex)
                swings = self._get_window_extremes_as_swings(df, tf)
                analysis = self.determine_structure(swings, tf)
                results[tf] = analysis

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

        # Спец-предобработка для 15M: ограничение окна и клип аномалий по ATR
        if timeframe == "15M" and len(data) > 0:
            try:
                # 1) Ограничим окно последними 7 днями
                if hasattr(data.index, "tz"):
                    cutoff = data.index[-1] - pd.Timedelta(days=7)
                else:
                    cutoff = pd.to_datetime(data.index[-1]) - pd.Timedelta(days=7)
                data = data.loc[data.index >= cutoff]
                # 2) Клип выбросов: High/Low за пределами ±8*ATR относительно Close
                high = data["High"].astype(float)
                low = data["Low"].astype(float)
                close = data["Close"].astype(float)
                tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
                atr_series = pd.Series(tr).rolling(window=14, min_periods=1).mean().fillna(method="bfill")
                upper = close + atr_series * 8.0
                lower = close - atr_series * 8.0
                # Применяем клип к копии, чтобы не трогать исходную ссылку
                data = data.copy()
                data["High"] = np.minimum(high.values, upper.values)
                data["Low"] = np.maximum(low.values, lower.values)
            except Exception:
                # В случае проблем с индексом/данными — пропускаем предобработку
                pass

        # Логирование входных экстремумов окна
        try:
            hi_series = data["High"].astype(float)
            lo_series = data["Low"].astype(float)
            self.logger.info(
                "SWING DETECTION START | tf=%s | %s → %s | points=%d | actual_high=%.2f (%s) | actual_low=%.2f (%s)",
                timeframe,
                str(data.index[0]),
                str(data.index[-1]),
                len(data),
                float(hi_series.max()),
                str(hi_series.idxmax()),
                float(lo_series.min()),
                str(lo_series.idxmin()),
            )
        except Exception:
            pass

        swing_points: List[SwingPoint] = []
        # Динамический размер окна: не больше предустановленного и не больше 10% длины ряда
        preset = self.swing_windows[timeframe]
        dyn = max(1, len(data) // 10)
        window = max(1, min(preset, dyn))
        # Если окно слишком велико для набора данных — используем экстремумы окна как swings
        if window * 2 >= len(data):
            # Вернём оба экстремума окна как swings (high и low)
            return self._get_window_extremes_as_swings(data, timeframe)

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

        # Гарантируем минимум по одному high и low: добавим экстремумы окна при нехватке
        highs_now = [s for s in swing_points if s.type == "high"]
        lows_now = [s for s in swing_points if s.type == "low"]
        if len(highs_now) == 0 or len(lows_now) == 0:
            hi_series2 = data["High"].astype(float)
            lo_series2 = data["Low"].astype(float)
            if len(highs_now) == 0:
                hi_idx = hi_series2.idxmax()
                swing_points.append(
                    SwingPoint(timestamp=hi_idx, price=float(hi_series2.max()), type="high", timeframe=timeframe, strength=1.0)
                )
            if len(lows_now) == 0:
                lo_idx = lo_series2.idxmin()
                swing_points.append(
                    SwingPoint(timestamp=lo_idx, price=float(lo_series2.min()), type="low", timeframe=timeframe, strength=1.0)
                )
            swing_points.sort(key=lambda x: x.timestamp)

        # Логирование результатов
        try:
            highs = [s for s in swing_points if s.type == "high"]
            lows = [s for s in swing_points if s.type == "low"]
            self.logger.info(
                "SWING DETECTION RESULT | tf=%s | highs=%d lows=%d | last_high=%s | last_low=%s",
                timeframe,
                len(highs),
                len(lows),
                f"{highs[-1].price:.2f}" if highs else "None",
                f"{lows[-1].price:.2f}" if lows else "None",
            )
        except Exception:
            pass
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

    def _validate_and_correct(self, data: pd.DataFrame, analysis: StructureAnalysis) -> StructureAnalysis:
        """Строгая валидация: swing уровни должны присутствовать в OHLC текущего окна.
        Если нет — корректируем на экстремумы окна, чтобы исключить "устаревшие" значения."""
        # Строгий допуск: ±10 пунктов
        strict_tol = 10.0
        hi_series = data["High"].astype(float)
        lo_series = data["Low"].astype(float)
        sh = float(analysis.last_swing_high.price)
        sl = float(analysis.last_swing_low.price)
        hi_ok = ((hi_series >= sh - strict_tol) & (hi_series <= sh + strict_tol)).any()
        lo_ok = ((lo_series >= sl - strict_tol) & (lo_series <= sl + strict_tol)).any()
        if hi_ok and lo_ok:
            return analysis

        # Критическая ошибка — принудительная замена
        self.logger.error(
            "КРИТИЧЕСКАЯ ОШИБКА SWING DETECTION | tf=%s | swing_high=%.2f exists=%s | swing_low=%.2f exists=%s | "
            "data_range: %s → %s | actual_high=%.2f actual_low=%.2f",
            analysis.timeframe,
            sh,
            str(hi_ok),
            sl,
            str(lo_ok),
            str(data.index[0]),
            str(data.index[-1]),
            float(hi_series.max()),
            float(lo_series.min()),
        )
        return self._force_use_window_extremes(data, analysis)

    def _force_use_window_extremes(self, data: pd.DataFrame, analysis: StructureAnalysis) -> StructureAnalysis:
        hi_series = data["High"].astype(float)
        lo_series = data["Low"].astype(float)
        actual_high = float(hi_series.max())
        actual_low = float(lo_series.min())
        actual_high_date = hi_series.idxmax()
        actual_low_date = lo_series.idxmin()

        sp_hi = SwingPoint(timestamp=actual_high_date, price=actual_high, type="high", timeframe=analysis.timeframe, strength=0.9)
        sp_lo = SwingPoint(timestamp=actual_low_date, price=actual_low, type="low", timeframe=analysis.timeframe, strength=0.9)

        self.logger.warning(
            "SWING CORRECTION APPLIED | tf=%s | old_high=%.2f → new_high=%.2f | old_low=%.2f → new_low=%.2f",
            analysis.timeframe,
            float(analysis.last_swing_high.price),
            actual_high,
            float(analysis.last_swing_low.price),
            actual_low,
        )

        if actual_high_date > actual_low_date:
            direction = StructureDirection.BULLISH
            break_level = actual_low
        else:
            direction = StructureDirection.BEARISH
            break_level = actual_high

        return StructureAnalysis(
            timeframe=analysis.timeframe,
            direction=direction,
            last_swing_high=sp_hi,
            last_swing_low=sp_lo,
            structure_strength=0.8,
            break_level=float(break_level),
            confidence=0.9,
        )

    def _get_window_extremes_as_swings(self, data: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
        if data is None or len(data) == 0:
            return []
        hi_series = data["High"].astype(float)
        lo_series = data["Low"].astype(float)
        high_idx = hi_series.idxmax()
        low_idx = lo_series.idxmin()
        swing_high = SwingPoint(timestamp=high_idx, price=float(hi_series.max()), type="high", timeframe=timeframe, strength=1.0)
        swing_low = SwingPoint(timestamp=low_idx, price=float(lo_series.min()), type="low", timeframe=timeframe, strength=1.0)
        return [swing_high, swing_low]

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


