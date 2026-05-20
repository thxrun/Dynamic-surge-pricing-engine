"""
generate_dataset.py
--------------------
Generates a realistic synthetic surge pricing dataset and saves it
to data/raw/surge_pricing.csv.

Features generated:
  - rider_count        : number of ride requests in the zone
  - driver_count       : number of active drivers in the zone
  - traffic_index      : road congestion level (1.0 = free flow, 5.0 = gridlock)
  - weather_severity   : weather impact score (0 = clear, 3 = severe storm)
  - hour               : hour of the day (0-23)
  - is_weekend         : 1 if Saturday/Sunday, else 0
  - demand_supply_ratio: rider_count / driver_count (key surge trigger)
  - surge_multiplier   : target label — how much to multiply base fare

Run:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES = 10_000
OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "surge_pricing.csv")


def generate_surge_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Build a synthetic surge pricing DataFrame with realistic correlations.

    Returns
    -------
    pd.DataFrame
        Raw dataset ready to be saved.
    """

    # 1. Time features
    hour = np.random.randint(0, 24, size=n_samples)
    is_weekend = np.random.choice([0, 1], size=n_samples, p=[0.71, 0.29])

    # 2. Peak-hour multiplier boosts rider demand
    #    Morning rush: 7-9 AM | Evening rush: 5-8 PM
    is_morning_peak = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_peak = ((hour >= 17) & (hour <= 20)).astype(int)
    peak_boost = 1.0 + 0.5 * is_morning_peak + 0.6 * is_evening_peak + 0.3 * is_weekend

    # 3. Rider count — higher during peaks and weekends
    base_riders = np.random.poisson(lam=40, size=n_samples)
    rider_count = np.clip((base_riders * peak_boost).astype(int), 5, 200)

    # 4. Driver count — inversely correlated with peak demand (drivers get busy)
    base_drivers = np.random.poisson(lam=35, size=n_samples)
    driver_availability = 1.0 - 0.3 * is_morning_peak - 0.35 * is_evening_peak
    driver_count = np.clip(
        (base_drivers * driver_availability).astype(int), 3, 150
    )

    # 5. Demand-supply ratio — the primary surge trigger
    demand_supply_ratio = np.round(rider_count / np.maximum(driver_count, 1), 3)

    # 6. Traffic index (1.0=clear, 5.0=gridlock) — worsens during peaks
    traffic_base = np.random.uniform(1.0, 3.0, size=n_samples)
    traffic_index = np.round(
        np.clip(traffic_base + 0.8 * is_morning_peak + 1.0 * is_evening_peak, 1.0, 5.0),
        2,
    )

    # 7. Weather severity (0=clear, 1=light rain, 2=heavy rain, 3=storm)
    weather_severity = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.60, 0.25, 0.10, 0.05])

    # 8. Surge multiplier (label) — deterministic formula + noise
    #    Business rule: surge = f(demand/supply, traffic, weather)
    surge_raw = (
        1.0
        + 0.4 * np.clip(demand_supply_ratio - 1.0, 0, None)  # kicks in above 1:1 ratio
        + 0.15 * (traffic_index - 1.0)                        # congestion premium
        + 0.10 * weather_severity                              # weather premium
        + np.random.normal(0, 0.05, size=n_samples)           # realistic noise
    )
    # Round to nearest 0.1x; floor at 1.0x (no discount), cap at 4.0x
    surge_multiplier = np.round(np.clip(surge_raw, 1.0, 4.0), 1)

    df = pd.DataFrame(
        {
            "rider_count": rider_count,
            "driver_count": driver_count,
            "traffic_index": traffic_index,
            "weather_severity": weather_severity,
            "hour": hour,
            "is_weekend": is_weekend,
            "demand_supply_ratio": demand_supply_ratio,
            "surge_multiplier": surge_multiplier,
        }
    )
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Generating {N_SAMPLES:,} synthetic surge pricing samples …")
    df = generate_surge_dataset(N_SAMPLES)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Dataset saved → {OUTPUT_FILE}")
    print(f"[INFO] Shape : {df.shape}")
    print(f"\n[INFO] Sample statistics:")
    print(df.describe().round(3).to_string())
    print(f"\n[INFO] Surge multiplier distribution:")
    print(df["surge_multiplier"].value_counts().sort_index().head(10))


if __name__ == "__main__":
    main()
