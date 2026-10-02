#!/usr/bin/env python3
"""Create an offline, synthetic taxi dataset for the quick demo."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000, help="число строк (по умолчанию 20 000)")
    parser.add_argument("--output", default="data/nyc_taxi_demo.parquet", help="путь CSV или Parquet")
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("нужно минимум 100 строк")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    rng = np.random.default_rng(24)
    start = pd.Timestamp("2024-01-01")
    rows = args.rows
    pickup = start + pd.to_timedelta(rng.integers(0, 31 * 24 * 60, rows), unit="m")
    distance = np.clip(rng.gamma(shape=2.2, scale=2.1, size=rows), 0.2, 35).round(2)
    passengers = rng.choice([1, 2, 3, 4, 5, 6], rows, p=[.64, .22, .07, .04, .02, .01])
    rush = np.isin(pickup.hour, [7, 8, 9, 16, 17, 18, 19]).astype(float)
    fare = np.maximum(3.5, 2.5 + distance * 2.65 + rush * 1.1 + rng.normal(0, 2.3, rows)).round(2)
    duration = np.maximum(3, (distance / rng.uniform(7, 18, rows) * 60 + rng.normal(4, 3, rows))).astype(int)
    tip = np.where(rng.random(rows) < .7, fare * rng.uniform(.08, .24, rows), 0).round(2)
    extra = rng.choice([0.0, 0.5, 1.0, 2.5], rows, p=[.45, .2, .15, .2])

    frame = pd.DataFrame({
        "VendorID": rng.choice([1, 2], rows),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration, unit="m"),
        "passenger_count": passengers,
        "trip_distance": distance,
        "RatecodeID": rng.choice([1, 2, 3, 4, 5, 6], rows, p=[.94, .02, .015, .008, .01, .007]),
        "store_and_fwd_flag": rng.choice(["N", "Y"], rows, p=[.98, .02]),
        "PULocationID": rng.integers(1, 266, rows),
        "DOLocationID": rng.integers(1, 266, rows),
        "payment_type": rng.choice([1, 2, 3, 4, 5, 6], rows, p=[.52, .42, .035, .02, .003, .002]),
        "fare_amount": fare,
        "extra": extra,
        "mta_tax": np.where(fare > 0, .5, 0),
        "tip_amount": tip,
        "tolls_amount": np.where(distance > 8, rng.choice([0.0, 6.94], rows), 0.0),
        "improvement_surcharge": np.where(fare > 0, .3, 0),
        "total_amount": fare + tip + extra + .8,
        "congestion_surcharge": rng.choice([0.0, 2.5], rows, p=[.25, .75]),
        "Airport_fee": rng.choice([0.0, 1.75], rows, p=[.94, .06]),
    })
    frame.loc[rng.random(rows) < .02, "passenger_count"] = np.nan
    frame.loc[rng.random(rows) < .01, "trip_distance"] = np.nan
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False, float_format="%.2f")
    elif output.suffix.lower() == ".parquet":
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), output, compression="zstd")
    else:
        parser.error("расширение файла должно быть .csv или .parquet")
    print("Демо-набор сохранён:", output, "—", rows, "синтетических строк")


if __name__ == "__main__":
    main()
