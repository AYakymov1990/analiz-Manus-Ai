import os
import json
import pytest

from src.core.strategy_engine import TradingAnalyzer


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Интеграционный тест требует сети; включить с RUN_INTEGRATION_TESTS=1",
)
def test_trading_analyzer_full_chain(tmp_path):
    analyzer = TradingAnalyzer()
    ctx = analyzer.analyze("GC=F", save_output=True)

    # базовые проверки структуры
    assert ctx["metadata"]["symbol"] == "GC=F"
    assert "market_structure" in ctx
    assert "fibonacci_analysis" in ctx
    assert "retail_behavior" in ctx
    assert "trading_setup" in ctx
    assert "manipulation_context" in ctx

    # проверка сохранения (хотя имя с TS, проверим папку на наличие хоть одного manus_ai_context файла)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "real_chain"))
    files = [f for f in os.listdir(out_dir) if f.startswith("manus_ai_context_") and f.endswith(".json")]
    assert len(files) >= 1


