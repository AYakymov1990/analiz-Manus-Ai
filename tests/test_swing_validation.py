import pandas as pd

from src.analyzers.structure_analyzer import MultiTimeframeStructureAnalyzer


def test_swing_detection_with_known_data():
    """Проверка: найденные свинги соответствуют экстремумам окна (±10)."""
    dates = pd.date_range('2025-09-01', periods=10, freq='D')
    test_data = pd.DataFrame({
        'High':  [3800, 3820, 3850, 3900, 3880, 3860, 3840, 3820, 3810, 3800],
        'Low':   [3780, 3800, 3830, 3870, 3860, 3840, 3820, 3800, 3790, 3780],
        'Close': [3790, 3810, 3840, 3890, 3870, 3850, 3830, 3810, 3800, 3790],
    }, index=dates)

    analyzer = MultiTimeframeStructureAnalyzer("TEST")
    swings = analyzer.detect_swing_points(test_data, "1D")

    # Должно быть минимум 2 swing точки
    assert len(swings) >= 2

    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    actual_max = test_data['High'].max()  # 3900
    actual_min = test_data['Low'].min()   # 3780

    found_max = max([s.price for s in highs]) if highs else 0
    found_min = min([s.price for s in lows]) if lows else float('inf')

    assert abs(found_max - actual_max) <= 10, f"Swing high {found_max} != actual {actual_max}"
    assert abs(found_min - actual_min) <= 10, f"Swing low {found_min} != actual {actual_min}"


