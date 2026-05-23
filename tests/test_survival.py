from routers.survival import (
    compute_hazard_rate, survival_curve, median_survival_days, generate_explanation
)


def test_hazard_rate_baseline():
    h = compute_hazard_rate(
        direction="up", rsi=55, volume_declining=False,
        near_band=False, sentiment_label="neutral",
        trend_age=5, regime="Bull"
    )
    assert 0.03 <= h <= 0.09


def test_hazard_rate_overbought_raises():
    h_neutral = compute_hazard_rate("up", 55, False, False, "neutral", 5, "Bull")
    h_overbought = compute_hazard_rate("up", 75, False, False, "neutral", 5, "Bull")
    assert h_overbought > h_neutral


def test_hazard_rate_bull_regime_lowers():
    h_bull = compute_hazard_rate("up", 55, False, False, "neutral", 5, "Bull")
    h_bear = compute_hazard_rate("up", 55, False, False, "neutral", 5, "Bear")
    assert h_bull < h_bear


def test_hazard_rate_old_trend_raises():
    h_young = compute_hazard_rate("up", 55, False, False, "neutral", 5, "Bull")
    h_old = compute_hazard_rate("up", 55, False, False, "neutral", 15, "Bull")
    assert h_old > h_young


def test_survival_curve_shape():
    curve = survival_curve(0.08)
    assert len(curve) == 7
    probs = [p["probability"] for p in curve]
    assert all(0 <= p <= 1 for p in probs)
    assert probs == sorted(probs, reverse=True)


def test_survival_curve_days():
    curve = survival_curve(0.08)
    assert [p["day"] for p in curve] == [1, 2, 3, 5, 7, 10, 14]


def test_median_survival_days_sanity():
    m_low = median_survival_days(0.04)
    m_high = median_survival_days(0.14)
    assert m_low > m_high
    assert m_high >= 1


def test_generate_explanation_returns_string():
    explanation = generate_explanation(
        direction="up", rsi=72, volume_declining=True,
        vol_decline_pct=23, near_upper_band=True, near_lower_band=False,
        sentiment_label="negative", trend_age=12, hazard_rate=0.10
    )
    assert isinstance(explanation, str)
    assert len(explanation) > 40
