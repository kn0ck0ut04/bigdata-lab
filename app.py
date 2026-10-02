from __future__ import annotations

import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "data" / "nyc_taxi_demo.parquet"
UPLOAD_DIR = ROOT / "instance" / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
CHUNK_ROWS = 100_000
# ponytail: train on a 20k reservoir sample to keep a Mac responsive; raise this cap if model quality requires more data.
SAMPLE_ROWS = 20_000
ALLOWED_EXTENSIONS = {".csv", ".parquet"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# ponytail: one active local dataset keeps the classroom app simple; add per-user jobs if it becomes multi-user.
ACTIVE_DATASET: Optional[Dict[str, Any]] = None
LAST_MODEL: Optional[Dict[str, Any]] = None


def iter_frames(path: Path) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False, encoding="utf-8-sig")
        return
    if path.suffix.lower() == ".parquet":
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=CHUNK_ROWS):
                yield batch.to_pandas()
        except pa.ArrowException as error:
            raise ValueError("Не удалось прочитать Parquet-файл: проверьте, что он не повреждён.") from error
        return
    raise ValueError("Поддерживаются файлы CSV и Parquet.")


def display_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
    result = str(value)
    return result if len(result) <= 72 else result[:69] + "…"


def target_options(sample: pd.DataFrame) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for name in sample.columns:
        column = sample[name].dropna()
        if len(column) < 2:
            continue
        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.notna().mean() >= 0.95:
            unique = numeric.nunique()
            if unique > 20:
                options.append({"name": str(name), "kind": "regression"})
            elif unique > 1:
                options.append({"name": str(name), "kind": "classification"})
        elif column.nunique() <= 20:
            options.append({"name": str(name), "kind": "classification"})
    return options


def histogram(sample: pd.DataFrame) -> Optional[Dict[str, Any]]:
    preferred = ("fare_amount", "trip_distance", "total_amount", "amount", "price")
    numeric = sample.select_dtypes(include=[np.number]).columns.tolist()
    column = next((name for token in preferred for name in numeric if token in str(name).lower()), None)
    if column is None:
        column = next((name for name in numeric if sample[name].nunique(dropna=True) > 10), None)
    if column is None:
        return None

    values = pd.to_numeric(sample[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    low, high = values.quantile([0.01, 0.99]).tolist()
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        low, high = float(values.min()), float(values.max())
    if low == high:
        return None
    counts, edges = np.histogram(values.clip(lower=low, upper=high), bins=10, range=(low, high))
    labels = ["{:.0f}–{:.0f}".format(edges[i], edges[i + 1]) for i in range(len(counts))]
    return {"title": str(column), "labels": labels, "values": counts.astype(int).tolist(), "sample_rows": int(len(values))}


def category_chart(sample: pd.DataFrame) -> Optional[Dict[str, Any]]:
    preferred = ("passenger_count", "payment_type", "vendorid", "ratecodeid")
    candidate = next(
        (name for token in preferred for name in sample.columns if token == str(name).lower()),
        None,
    )
    if candidate is None:
        for name in sample.columns:
            values = sample[name].dropna()
            if 2 <= values.nunique() <= 12 and (not pd.api.types.is_numeric_dtype(values) or values.nunique() <= 8):
                candidate = name
                break
    if candidate is None:
        return None
    counts = sample[candidate].fillna("Нет данных").astype(str).value_counts().head(8)
    return {
        "title": str(candidate),
        "labels": [str(label) for label in counts.index],
        "values": [int(value) for value in counts.values],
        "sample_rows": int(len(sample)),
    }


def analyze_file(path: Path, filename: str, size_bytes: int) -> Dict[str, Any]:
    started = time.monotonic()
    rng = np.random.default_rng(42)
    sample: Optional[pd.DataFrame] = None
    sample_keys = np.array([], dtype=float)
    rows = 0
    columns: List[str] = []
    missing: Dict[str, int] = {}

    for frame in iter_frames(path):
        if frame.empty:
            continue
        if not columns:
            columns = [str(name) for name in frame.columns]
            if len(columns) > 200:
                raise ValueError("Слишком много столбцов: сейчас поддерживается до 200.")
            missing = {name: 0 for name in columns}
        if len(frame.columns) != len(columns):
            raise ValueError("В разных частях файла обнаружено разное число столбцов.")

        rows += len(frame)
        for name, count in frame.isna().sum().items():
            key = str(name)
            missing[key] = missing.get(key, 0) + int(count)

        keys = rng.random(len(frame))
        if sample is None:
            combined, combined_keys = frame.reset_index(drop=True), keys
        else:
            combined = pd.concat([sample, frame], ignore_index=True)
            combined_keys = np.concatenate([sample_keys, keys])
        if len(combined) > SAMPLE_ROWS:
            selected = np.argpartition(combined_keys, -SAMPLE_ROWS)[-SAMPLE_ROWS:]
            sample, sample_keys = combined.iloc[selected].reset_index(drop=True), combined_keys[selected]
        else:
            sample, sample_keys = combined, combined_keys

    if rows == 0 or sample is None:
        raise ValueError("Файл пустой или не содержит строк с данными.")
    sample.columns = [str(name) for name in sample.columns]
    profiles = []
    for name in columns:
        series = sample[name]
        profiles.append({
            "name": name,
            "type": str(series.dtype),
            "missing": missing.get(name, 0),
            "unique_sample": int(series.nunique(dropna=True)),
        })

    preview = [
        {str(name): display_value(value) for name, value in row.items()}
        for row in sample.head(8).to_dict(orient="records")
    ]
    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "rows": rows,
        "column_count": len(columns),
        "missing_cells": sum(missing.values()),
        "sample_rows": len(sample),
        "seconds": round(time.monotonic() - started, 1),
        "columns": profiles,
        "targets": target_options(sample),
        "preview": preview,
        "distribution": histogram(sample),
        "categories": category_chart(sample),
    }, sample


def prepare_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    features = frame.copy()
    date_tokens = ("date", "time", "datetime")
    for name in list(features.columns):
        series = features[name]
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        looks_like_date = any(token in str(name).lower() for token in date_tokens)
        if is_datetime or (looks_like_date and not pd.api.types.is_numeric_dtype(series)):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.5:
                features[str(name) + " hour"] = parsed.dt.hour
                features[str(name) + " weekday"] = parsed.dt.dayofweek
                features[str(name) + " month"] = parsed.dt.month
            features = features.drop(columns=[name])

    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [name for name in features.columns if name not in numeric]
    for name in numeric:
        features[name] = pd.to_numeric(features[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        features[name] = features[name].astype("string").fillna("Нет данных").astype(str)
    return features, numeric, categorical


def train_model(target_name: str) -> Dict[str, Any]:
    global LAST_MODEL
    if ACTIVE_DATASET is None:
        raise ValueError("Сначала загрузите набор данных.")
    sample = ACTIVE_DATASET["sample"]
    if target_name not in sample.columns:
        raise ValueError("Выбранный целевой столбец не найден в данных.")

    usable = sample.dropna(subset=[target_name]).copy()
    target = usable.pop(target_name)
    numeric_target = pd.to_numeric(target, errors="coerce")
    is_numeric = numeric_target.notna().mean() >= 0.95
    if is_numeric:
        target = numeric_target
        usable = usable.loc[target.notna()]
        target = target.loc[usable.index]
    elif target.nunique() > 20:
        raise ValueError("Для обучения выберите числовой столбец или категорию не более чем с 20 значениями.")
    if target.nunique() < 2:
        raise ValueError("Для обучения нужно минимум два разных значения целевого столбца.")

    if target_name.lower() == "fare_amount":
        leakage_columns = {
            "total_amount", "tip_amount", "extra", "mta_tax", "tolls_amount",
            "improvement_surcharge", "congestion_surcharge", "airport_fee",
        }
        usable = usable.drop(columns=[name for name in usable.columns if str(name).lower() in leakage_columns])

    features, numeric_columns, categorical_columns = prepare_features(usable)
    keep = [name for name in features.columns if features[name].nunique(dropna=True) > 1]
    features = features[keep]
    numeric_columns = [name for name in numeric_columns if name in keep]
    categorical_columns = [name for name in categorical_columns if name in keep]
    if not keep:
        raise ValueError("В данных нет признаков для обучения модели.")

    task = "regression" if is_numeric and target.nunique() > 20 else "classification"
    if task == "classification":
        target = target.astype(str)
    else:
        target = target.astype(float)

    stratify = None
    if task == "classification":
        counts = target.value_counts()
        test_rows = max(1, int(len(target) * 0.2))
        if counts.min() >= 2 and target.nunique() <= test_rows:
            stratify = target
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_columns))
    if categorical_columns:
        encoder = OneHotEncoder(handle_unknown="ignore", max_categories=30)
        transformers.append(("categorical", encoder, categorical_columns))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    estimator = (
        RandomForestRegressor(n_estimators=70, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=42)
        if task == "regression"
        else RandomForestClassifier(n_estimators=70, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=42)
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    if task == "regression":
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "rmse": round(float(math.sqrt(mean_squared_error(y_test, predictions))), 3),
            "r2": round(float(r2_score(y_test, predictions)), 3),
        }
        task_name = "Регрессия"
    else:
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 3),
            "classes": int(target.nunique()),
        }
        task_name = "Классификация"

    importances = pipeline.named_steps["model"].feature_importances_
    transformed = pipeline.named_steps["preprocessor"].get_feature_names_out()
    grouped: Dict[str, float] = {}
    all_features = sorted(keep, key=len, reverse=True)
    for feature_name, importance in zip(transformed, importances):
        clean_name = feature_name.split("__", 1)[-1]
        original = next((name for name in all_features if clean_name == name or clean_name.startswith(name + "_")), clean_name)
        grouped[original] = grouped.get(original, 0.0) + float(importance)
    top_features = [
        {"name": name, "value": round(value, 4)}
        for name, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:6]
    ]
    LAST_MODEL = {
        "target": target_name,
        "task": task_name,
        "metrics": metrics,
        "rows": int(len(target)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": top_features,
        "distribution": ACTIVE_DATASET["summary"]["distribution"],
    }
    return LAST_MODEL


def public_state() -> Dict[str, Any]:
    return {
        "dataset": ACTIVE_DATASET["summary"] if ACTIVE_DATASET else None,
        "model": LAST_MODEL,
    }


def load_into_memory(path: Path, filename: str, size_bytes: int) -> Dict[str, Any]:
    global ACTIVE_DATASET, LAST_MODEL
    summary, sample = analyze_file(path, filename, size_bytes)
    ACTIVE_DATASET = {"path": path, "sample": sample, "summary": summary}
    LAST_MODEL = None
    return summary


def save_upload() -> Dict[str, Any]:
    file = request.files.get("file")
    if file is None or not file.filename:
        raise ValueError("Выберите CSV- или Parquet-файл.")
    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Поддерживаются только CSV и Parquet.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / (uuid.uuid4().hex + extension)
    file.save(str(destination))
    try:
        return load_into_memory(destination, filename, destination.stat().st_size)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


@app.get("/")
def index() -> str:
    return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024))


@app.post("/upload")
def upload() -> Any:
    try:
        summary = save_upload()
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(error)}), 400
        return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), error=str(error)), 400
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"dataset": summary, "model": None})
    return redirect(url_for("index"))


@app.post("/sample")
def load_sample() -> Any:
    if not SAMPLE_FILE.exists():
        message = "Демо-данные не найдены. Скачайте официальный файл командой из README.md."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": message}), 404
        return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), error=message), 404
    try:
        summary = load_into_memory(SAMPLE_FILE, "nyc_taxi_demo.parquet", SAMPLE_FILE.stat().st_size)
    except Exception as error:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(error)}), 400
        return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), error=str(error)), 400
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"dataset": summary, "model": None})
    return redirect(url_for("index"))


@app.post("/train")
def train() -> Any:
    target = request.form.get("target", "")
    try:
        result = train_model(target)
    except (ValueError, KeyError) as error:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(error)}), 400
        return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), error=str(error)), 400
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"model": result})
    return redirect(url_for("index"))


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_: RequestEntityTooLarge) -> Any:
    message = "Файл больше лимита. Измените MAX_UPLOAD_MB перед запуском приложения."
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"error": message}), 413
    return render_template("index.html", state=public_state(), max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024), error=message), 413


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
