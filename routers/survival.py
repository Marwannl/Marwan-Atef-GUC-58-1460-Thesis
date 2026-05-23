DAYS = [1, 2, 3, 5, 7, 10, 14]
BASE_HAZARD = 0.06


def compute_hazard_rate(direction: str, rsi: float, volume_declining: bool,
                        near_band: bool, sentiment_label: str,
                        trend_age: int, regime: str) -> float:
    h = BASE_HAZARD

    if direction == "up":
        if rsi > 70:
            h += 0.04
        elif 50 <= rsi <= 60:
            h -= 0.01
        if sentiment_label == "negative":
            h += 0.02
        if regime == "Bull":
            h -= 0.02
    else:
        if rsi < 30:
            h += 0.04
        elif 40 <= rsi <= 50:
            h -= 0.01
        if sentiment_label == "positive":
            h += 0.02
        if regime == "Bear":
            h -= 0.02

    if volume_declining:
        h += 0.03
    if near_band:
        h += 0.03
    if trend_age > 20:
        h += 0.05
    elif trend_age > 10:
        h += 0.02

    return round(max(0.01, min(h, 0.40)), 6)


def survival_curve(hazard_rate: float) -> list:
    return [
        {"day": d, "probability": round((1 - hazard_rate) ** d, 4)}
        for d in DAYS
    ]


def median_survival_days(hazard_rate: float) -> int:
    for d in range(1, 60):
        if (1 - hazard_rate) ** d < 0.5:
            return d
    return 60


def generate_explanation(direction: str, rsi: float, volume_declining: bool,
                         vol_decline_pct: float, near_upper_band: bool,
                         near_lower_band: bool, sentiment_label: str,
                         trend_age: int, hazard_rate: float) -> str:
    trend_label = "uptrend" if direction == "up" else "downtrend"
    parts = []

    if direction == "up":
        if rsi > 70:
            parts.append(f"RSI at {rsi:.0f} is in overbought territory")
        elif rsi > 62:
            parts.append(f"RSI at {rsi:.0f} is approaching overbought territory")
        if near_upper_band:
            parts.append("price is pressing against the upper Bollinger Band")
        if sentiment_label == "negative":
            parts.append("sentiment has shifted negative after recent headlines")
    else:
        if rsi < 30:
            parts.append(f"RSI at {rsi:.0f} is in oversold territory")
        elif rsi < 38:
            parts.append(f"RSI at {rsi:.0f} is approaching oversold territory")
        if near_lower_band:
            parts.append("price is pressing against the lower Bollinger Band")
        if sentiment_label == "positive":
            parts.append("sentiment is positive despite the downtrend, suggesting potential reversal")

    if volume_declining and vol_decline_pct > 5:
        parts.append(
            f"volume has declined {vol_decline_pct:.0f}% over the last 3 days "
            f"suggesting weakening conviction"
        )

    if trend_age > 20:
        parts.append(
            f"at {trend_age} days old this is a mature trend, "
            f"statistical fragility increases past 20 days"
        )
    elif trend_age > 10:
        parts.append(f"the trend has been running {trend_age} days, older trends tend to be more fragile")

    survival_7d = round((1 - hazard_rate) ** 7 * 100)

    if not parts:
        body = "technical indicators show no strong reversal signals, this trend appears healthy"
    elif len(parts) == 1:
        body = parts[0].capitalize()
    else:
        body = ", ".join(parts[:-1]).capitalize() + f", and {parts[-1]}"

    exhaustion = "showing early signs of exhaustion" if hazard_rate > 0.08 else "holding up well"
    return (
        f"This {trend_label} is {exhaustion}. "
        f"{body}. "
        f"The survival model gives this trend a {survival_7d}% chance of surviving the next 7 days."
    )
