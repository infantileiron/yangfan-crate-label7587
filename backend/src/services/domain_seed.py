"""Reproducible simulated crate-cycle logs for Chengwan's wash bay."""

from datetime import date, timedelta
from math import exp
from random import Random


SEED = 20261125


def sample_rows() -> list[dict]:
    rng = Random(SEED)
    rows = []
    first_day = date(2026, 11, 17)
    for offset in range(8):
        day = first_day + timedelta(days=offset)
        for shift in ("A", "B"):
            batch_id = f"CW-{day:%m%d}-{shift}"
            for cycle in range(1, 17):
                pressure = round(rng.uniform(3.0, 8.0), 2)
                current = round(3.0 + 1.15 * pressure + rng.uniform(-0.22, 0.22), 2)
                temperature = round(rng.uniform(44, 80), 1)
                dewpoint = round(rng.uniform(1, 24), 1)
                dwell = round(rng.uniform(18, 100), 1)
                conductivity = round(rng.uniform(0.65, 2.25), 2)
                # Strong jets and hot or cold rinse extremes increase detachment risk.
                # A humid dryer and short cure time add smaller independent effects.
                risk = (-2.1 + 2.05 * (pressure - 5.5) + 0.020 * (temperature - 62) ** 2
                        + 0.075 * (dewpoint - 12) - 0.021 * (dwell - 55))
                detached = int(rng.random() < 1 / (1 + exp(-risk)))
                rows.append({
                    "cycle_id": f"{batch_id}-{cycle:02}",
                    "batch_id": batch_id,
                    "wash_date": day.isoformat(),
                    "detached": detached,
                    "spray_pressure_bar": pressure,
                    "pump_current_a": current,
                    "rinse_temp_c": temperature,
                    "dryer_dewpoint_c": dewpoint,
                    "label_dwell_min": dwell,
                    "conductivity_ms_cm": conductivity,
                })
    return rows
