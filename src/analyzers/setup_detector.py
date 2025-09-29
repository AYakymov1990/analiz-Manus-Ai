from dataclasses import dataclass
from typing import Dict, Any

from ..core.data_structures import SetupType


@dataclass
class SetupResult:
    setup_type: SetupType
    confidence: float
    conditions: Dict[str, Any]


class SetupDetector:
    def _analyze_timeframe_alignment(self, structures, fibs) -> Dict[str, Any]:
        s1d = structures["1D"].direction.value
        s4h = structures["4H"].direction.value
        f1d = fibs["1D"].current_zone.value
        f4h = fibs["4H"].current_zone.value
        aligned = (s1d == s4h)
        return {"s1d": s1d, "s4h": s4h, "f1d": f1d, "f4h": f4h, "aligned": aligned}

    def _assess_retail_vulnerability(self, retail, fibs) -> Dict[str, Any]:
        # Простая эвристика: extension => FOMO, discount => buy interest
        f1d_zone = fibs["1D"].current_zone.value
        sentiment = "fomo_buying" if f1d_zone.startswith("extension") else (
            "buy_interest" if f1d_zone in ("discount", "ote") else "neutral"
        )
        has_ssl = any(z.zone_type == "SSL" for z in retail["liquidity_zones"]) if isinstance(retail, dict) else False
        return {"sentiment": sentiment, "has_ssl": has_ssl}

    def _analyze_extension_scenarios(self, fibs) -> Dict[str, Any]:
        ext_1d = fibs["1D"].current_zone.value.startswith("extension")
        ext_4h = fibs["4H"].current_zone.value.startswith("extension")
        return {"extension_1d": ext_1d, "extension_4h": ext_4h, "any_extension": ext_1d or ext_4h}

    def _calculate_setup_confidence(self, conditions: Dict[str, bool]) -> float:
        if not conditions:
            return 0.0
        score = sum(1 for v in conditions.values() if bool(v))
        return min(1.0, max(0.0, score / len(conditions)))

    def _classify_setup(self, alignment: Dict[str, Any], retail_vuln: Dict[str, Any], ext: Dict[str, Any]) -> SetupResult:
        # Extension Reversal
        if ext["any_extension"] and retail_vuln["sentiment"] == "fomo_buying":
            cond = {
                "any_extension": ext["any_extension"],
                "fomo": True,
                "aligned_bullish": alignment["aligned"] and alignment["s1d"] == "bullish",
            }
            return SetupResult(SetupType.EXTENSION_REVERSAL, self._calculate_setup_confidence(cond), cond)

        # Counter-trend SSL hunt (1D bullish + H4 OTE/Premium + SSL)
        cond_ssl = {
            "s1d_bullish": alignment["s1d"] == "bullish",
            "h4_ote_premium": alignment["f4h"] in ("ote", "premium"),
            "ssl_present": retail_vuln["has_ssl"],
        }
        if all(cond_ssl.values()):
            return SetupResult(SetupType.COUNTER_TREND_SSL_HUNT, self._calculate_setup_confidence(cond_ssl), cond_ssl)

        # Trend continuation (alignment и H4 premium)
        cond_tc = {
            "aligned": alignment["aligned"],
            "h4_premium": alignment["f4h"] == "premium",
        }
        if all(cond_tc.values()):
            return SetupResult(SetupType.TREND_CONTINUATION, self._calculate_setup_confidence(cond_tc), cond_tc)

        # Structure break wait (конфликт направлений)
        cond_sb = {"conflict": not alignment["aligned"]}
        if cond_sb["conflict"]:
            return SetupResult(SetupType.STRUCTURE_BREAK_WAIT, self._calculate_setup_confidence(cond_sb), cond_sb)

        return SetupResult(SetupType.NO_SETUP, 0.0, {"reason": "no rules matched"})

    def detect_setup(self, structure_data, fibonacci_data, retail_data) -> SetupResult:
        alignment = self._analyze_timeframe_alignment(structure_data, fibonacci_data)
        retail_vuln = self._assess_retail_vulnerability(retail_data, fibonacci_data)
        extension_context = self._analyze_extension_scenarios(fibonacci_data)
        return self._classify_setup(alignment, retail_vuln, extension_context)


