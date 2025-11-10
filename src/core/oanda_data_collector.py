import os
import logging
from typing import Optional

import requests
import pandas as pd


class OANDADataCollector:
    """
    OANDA Data Collector for Smart Money Concepts Trading System.
    Provides OHLC data compatible with existing analysis pipeline.
    """

    SYMBOL_MAP = {
        # Forex
        "EURUSD=X": "EUR_USD",
        "EURUSD": "EUR_USD",
        "EUR/USD": "EUR_USD",
        "GBPUSD=X": "GBP_USD",
        "GBPUSD": "GBP_USD",
        "GBP/USD": "GBP_USD",
        "USDJPY=X": "USD_JPY",
        "USDJPY": "USD_JPY",
        # Commodities
        "GC=F": "XAU_USD",
        "XAUUSD": "XAU_USD",
        "GOLD": "XAU_USD",
        "XAU/USD": "XAU_USD",
        # Indices
        "^GDAXI": "DE30_EUR",
        "GER40": "DE30_EUR",
        "DAX": "DE30_EUR",
        "DE30": "DE30_EUR",
        "^NDX": "NAS100_USD",
        "NAS100": "NAS100_USD",
        "NASDAQ": "NAS100_USD",
        "MNQ=F": "NAS100_USD",
        # S&P
        "^GSPC": "SPX500_USD",
        "SPX500": "SPX500_USD",
        "SP500": "SPX500_USD",
        "SPX": "SPX500_USD",
        # Crypto
        "BTC-USD": "BTC_USD",
        "BTCUSD": "BTC_USD",
    }

    TIMEFRAME_MAP = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "M15": "M15",
        "30m": "M30",
        "1h": "H1",
        "60m": "H1",
        "4h": "H4",
        "H4": "H4",
        "1d": "D",
        "D": "D",
        "1D": "D",
    }

    def __init__(self, api_key: Optional[str] = None, account_type: str = "practice") -> None:
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        if not self.api_key:
            raise ValueError("OANDA_API_KEY is not set. Provide api_key or set env var OANDA_API_KEY.")
        self.account_type = account_type or os.getenv("OANDA_ACCOUNT_TYPE", "practice")
        self.base_url = "https://api-fxpractice.oanda.com" if self.account_type == "practice" else "https://api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.logger.info("OANDADataCollector initialized (mode=%s)", self.account_type)

    def _convert_symbol(self, symbol: str) -> str:
        oanda_symbol = self.SYMBOL_MAP.get(symbol.upper())
        if not oanda_symbol:
            self.logger.warning("Unknown symbol '%s', using as-is. Supported: %s", symbol, list(self.SYMBOL_MAP.keys()))
            return symbol
        return oanda_symbol

    def _convert_timeframe(self, tf: str) -> str:
        oanda_tf = self.TIMEFRAME_MAP.get(tf)
        if not oanda_tf:
            # try normalized lower-case
            oanda_tf = self.TIMEFRAME_MAP.get(tf.lower(), tf)
        return oanda_tf

    def get_candles(self, symbol: str, timeframe: str, count: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLC candles from OANDA. Returns DataFrame with columns:
        Open, High, Low, Close, Volume; index is timestamp (tz-aware).
        """
        oanda_symbol = self._convert_symbol(symbol)
        oanda_tf = self._convert_timeframe(timeframe)
        url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
        params = {
            "granularity": oanda_tf,
            "count": min(int(count), 5000),
            "price": "M",  # mid prices
        }
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to fetch data from OANDA: %s", str(e))
            raise RuntimeError(f"Failed to fetch data from OANDA: {e}")

        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", False):
                continue
            try:
                candles.append(
                    {
                        "timestamp": pd.to_datetime(c["time"]),
                        "Open": float(c["mid"]["o"]),
                        "High": float(c["mid"]["h"]),
                        "Low": float(c["mid"]["l"]),
                        "Close": float(c["mid"]["c"]),
                        "Volume": int(c.get("volume", 0)),
                    }
                )
            except (KeyError, ValueError) as e:
                self.logger.warning("Failed to parse candle: %s", str(e))
                continue

        df = pd.DataFrame(candles)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()
        self.logger.info("Loaded %d %s candles for %s", len(df), timeframe, symbol)
        return df

    def download(self, symbol: str, period: str = "59d", interval: str = "15m", **kwargs) -> pd.DataFrame:
        """
        Compatibility shim to emulate yfinance.download signature.
        Converts period to count approximately; uses get_candles under the hood.
        """
        # Approximate counts; conservative default
        # For 59 days of 15m: ~5664; limit to 1000 for performance
        count = int(kwargs.get("count", 1000))
        return self.get_candles(symbol, interval, count)


_global_collector: Optional[OANDADataCollector] = None


def download(symbol: str, period: str = "59d", interval: str = "15m", **kwargs) -> pd.DataFrame:
    """
    Global function to replace yf.download usage sites, if any remain.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = OANDADataCollector()
    return _global_collector.download(symbol, period, interval, **kwargs)


