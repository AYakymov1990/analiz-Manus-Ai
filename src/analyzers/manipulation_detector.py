from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from ..core.data_structures import ManipulationType


@dataclass
class ManipulationResult:
    manipulation_type: ManipulationType
    confidence: float
    details: Dict[str, Any]


class ManipulationDetector:
    def detect_manipulation(self, m15_data: pd.DataFrame, setup_context: Dict[str, Any]) -> ManipulationResult:
        if m15_data is None or len(m15_data) < 20:
            return ManipulationResult(ManipulationType.NO_MANIPULATION, 0.0, {"reason": "insufficient_data"})

        m15 = m15_data.copy()
        m15 = m15[["Open", "High", "Low", "Close"]].astype(float)

        structure_breaks = self._detect_structure_breaks(m15)
        volume_anomalies = self._analyze_volume_patterns(m15_data)
        momentum_spikes = self._detect_momentum_spikes(m15)

        manipulation = self._classify_manipulation(
            structure_breaks, volume_anomalies, momentum_spikes, setup_context
        )
        return manipulation

    def _detect_structure_breaks(self, data: pd.DataFrame) -> Dict[str, Any]:
        lows = data["Low"].values
        highs = data["High"].values
        idx = np.arange(len(data))

        recent_window = slice(max(0, len(data) - 120), len(data))
        recent_lows = lows[recent_window]
        recent_highs = highs[recent_window]

        swing_low = np.min(recent_lows)
        swing_high = np.max(recent_highs)

        broke_below = lows[-1] < swing_low
        broke_above = highs[-1] > swing_high

        lookback = min(8, len(data) - 1)
        returned_above = any(data["Close"].iloc[-k] > swing_low for k in range(1, lookback)) if broke_below else False
        returned_below = any(data["Close"].iloc[-k] < swing_high for k in range(1, lookback)) if broke_above else False

        # Доп. логика: недавний пробой предыдущего экстремума (исключая последние 3 свечи) с возвратом
        if len(data) >= 5:
            prev_high = float(np.max(highs[:-3])) if len(highs) > 3 else float(np.max(highs))
            prev_low = float(np.min(lows[:-3])) if len(lows) > 3 else float(np.min(lows))
            recent_break_above = any(h > prev_high for h in highs[-3:])
            recent_break_below = any(l < prev_low for l in lows[-3:])
            recent_return_below = float(data["Close"].iloc[-1]) < prev_high
            recent_return_above = float(data["Close"].iloc[-1]) > prev_low

            broke_above = broke_above or recent_break_above
            broke_below = broke_below or recent_break_below
            returned_below = returned_below or (recent_break_above and recent_return_below)
            returned_above = returned_above or (recent_break_below and recent_return_above)

        return {
            "swing_low": float(swing_low),
            "swing_high": float(swing_high),
            "broke_below": bool(broke_below),
            "broke_above": bool(broke_above),
            "returned_above": bool(returned_above),
            "returned_below": bool(returned_below),
        }

    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        if "Volume" not in data.columns:
            return {"volume_available": False, "spike": False, "zscore": 0.0}
        vol = data["Volume"].astype(float)
        if vol.isna().all():
            return {"volume_available": False, "spike": False, "zscore": 0.0}
        ma = vol.rolling(50, min_periods=10).mean()
        std = vol.rolling(50, min_periods=10).std().replace(0.0, np.nan)
        z = (vol - ma) / std
        z_last = float(z.iloc[-1]) if not np.isnan(z.iloc[-1]) else 0.0
        return {"volume_available": True, "spike": z_last > 2.0, "zscore": z_last}

    def _detect_momentum_spikes(self, data: pd.DataFrame) -> Dict[str, Any]:
        close = data["Close"].astype(float)
        ret = close.pct_change().fillna(0.0)
        ema_fast = close.ewm(span=5, adjust=False).mean()
        ema_slow = close.ewm(span=20, adjust=False).mean()
        ema_cross_up = bool((ema_fast.iloc[-1] > ema_slow.iloc[-1]) and (ema_fast.iloc[-2] <= ema_slow.iloc[-2]))
        ema_cross_down = bool((ema_fast.iloc[-1] < ema_slow.iloc[-1]) and (ema_fast.iloc[-2] >= ema_slow.iloc[-2]))
        spike = bool(abs(ret.iloc[-1]) > max(0.003, 3 * ret.rolling(50, min_periods=10).std().iloc[-1]))
        return {"ema_up": ema_cross_up, "ema_down": ema_cross_down, "spike": spike}

    def _classify_manipulation(
        self,
        structure_breaks: Dict[str, Any],
        volume_anomalies: Dict[str, Any],
        momentum_spikes: Dict[str, Any],
        setup_context: Dict[str, Any],
    ) -> ManipulationResult:
        score = 0.0
        details: Dict[str, Any] = {
            "structure": structure_breaks,
            "volume": volume_anomalies,
            "momentum": momentum_spikes,
        }

        # Контекст: зоны ликвидности и фибо
        liq = setup_context.get("liquidity_zones", {})
        bsl_zones = liq.get("BSL", [])
        ssl_zones = liq.get("SSL", [])
        current_price = float(setup_context.get("current_price", 0.0) or 0.0)
        fib_zone = str(setup_context.get("fibonacci_zone", ""))

        def near_any(levels, price, tol=20.0):
            try:
                return any(abs(float(level) - price) <= tol for level in levels)
            except Exception:
                return False

        near_bsl = near_any(bsl_zones, current_price)
        near_ssl = near_any(ssl_zones, current_price)
        in_extension = ("extension" in fib_zone)

        # Stop-hunt below
        if structure_breaks["broke_below"] and structure_breaks["returned_above"]:
            score += 0.6
            if momentum_spikes["spike"]:
                score += 0.2
            if volume_anomalies.get("spike", False):
                score += 0.1
            if near_ssl:
                score += 0.2
            if in_extension:
                score += 0.1
            return ManipulationResult(ManipulationType.STOP_HUNT_BELOW, min(1.0, score), details)

        # Stop-hunt above
        if structure_breaks["broke_above"] and structure_breaks["returned_below"]:
            score += 0.6
            if momentum_spikes["spike"]:
                score += 0.2
            if volume_anomalies.get("spike", False):
                score += 0.1
            if near_bsl:
                score += 0.2
            if in_extension:
                score += 0.1
            return ManipulationResult(ManipulationType.STOP_HUNT_ABOVE, min(1.0, score), details)

        # False breakout: пробой без подтверждения
        if structure_breaks["broke_below"] or structure_breaks["broke_above"]:
            if not momentum_spikes["spike"] and not volume_anomalies.get("spike", False):
                return ManipulationResult(ManipulationType.FALSE_BREAKOUT, 0.5, details)

        # Liquidity grab fallback
        if momentum_spikes["spike"] or volume_anomalies.get("spike", False):
            return ManipulationResult(ManipulationType.LIQUIDITY_GRAB, 0.4, details)

        return ManipulationResult(ManipulationType.NO_MANIPULATION, 0.0, details)


