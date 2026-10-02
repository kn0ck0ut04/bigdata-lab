#!/usr/bin/env python3
"""Download the public NYC TLC January 2024 taxi trip file."""

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FULL_PATH = DATA_DIR / "yellow_tripdata_2024-01.parquet"
DEMO_PATH = DATA_DIR / "nyc_taxi_demo.parquet"
SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"


def download(destination: Path) -> None:
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20",
        "--max-time", "300", "--speed-time", "45", "--speed-limit", "1024",
        "--output", str(destination), SOURCE_URL,
    ], check=True)
    print("Файл сохранён:", destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="сформировать 20 000 строк для быстрой демонстрации")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.demo:
        if not FULL_PATH.exists():
            download(FULL_PATH)
        print("Загрузите этот файл в приложении для обработки всего месяца.")
        return 0

    temporary = None
    source = FULL_PATH
    if not source.exists():
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        source = temporary
        download(source)
    try:
        parquet = pq.ParquetFile(source)
        batch = next(parquet.iter_batches(batch_size=20_000), None)
        if batch is None:
            raise ValueError("В исходном файле нет строк.")
        pq.write_table(batch, DEMO_PATH, compression="zstd")
        print("Демо-набор сохранён:", DEMO_PATH, "—", batch.num_rows, "строк")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Ошибка загрузки данных:", error, file=sys.stderr)
        raise SystemExit(1)
