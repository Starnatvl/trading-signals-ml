# src/api/inference.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Добавляем корень проекта в путь
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.features.feature_pipeline import add_features
from src.api.model_bundle import load_model_bundle

# Путь к модели (можно переопределить через переменную окружения)
MODEL_PATH = BASE_DIR / 'models' / 'prod_cat_seq.joblib'

_bundle = None  # кэш для загруженного bundle


def _load_bundle() -> dict[str, Any]:
    """Загружает и кэширует bundle модели."""
    global _bundle
    if _bundle is None:
        _bundle = load_model_bundle(str(MODEL_PATH))
    return _bundle


def _add_rolling_features(df_base: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Добавляет rolling mean/std для ключевых фичей по окнам.
    Имена создаются по шаблону: {feat}_roll{window}_{agg}
    """
    seq_key_feats = bundle.get('seq_key_feats')
    seq_windows = bundle.get('seq_windows')
    if seq_key_feats is None or seq_windows is None:
        raise ValueError("bundle должен содержать 'seq_key_feats' и 'seq_windows' для расчёта rolling-фичей")

    df = df_base.copy()
    for feat in seq_key_feats:
        for w in seq_windows:
            df[f"{feat}_roll{w}_mean"] = df.groupby("session_key")[feat].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{feat}_roll{w}_std"] = df.groupby("session_key")[feat].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    return df


def predict(window_df: pd.DataFrame) -> tuple[int, float]:
    """
    Принимает DataFrame с историей (минимум 60 строк) и возвращает (signal, confidence),
    где signal = 1 (BUY), -1 (SELL) или 0 (HOLD).
    confidence – вероятность класса 1 (BUY).
    """
    bundle = _load_bundle()

    df = window_df.copy()

    # Приводим имена колонок к ожидаемым
    if 'close' in df.columns and 'close_price' not in df.columns:
        df.rename(columns={'close': 'close_price'}, inplace=True)

    # Добавляем datetime, если нет
    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'datetime' not in df.columns:
        # Если нет timestamp, генерируем искусственно (не должно случаться в проде)
        df['datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1min')

    # Фиктивная сессия для группировок
    if 'session_key' not in df.columns:
        df['session_key'] = 'dummy'

    # 1. Рассчитываем базовые фичи через feature_pipeline
    df_base, _ = add_features(df, session_key_col='session_key')

    # Проверяем наличие всех базовых фичей, указанных в bundle['base_features']
    base_features = bundle.get('base_features')
    if base_features:
        missing_base = set(base_features) - set(df_base.columns)
        if missing_base:
            raise ValueError(f"Отсутствуют базовые фичи: {missing_base}")

    # 2. Добавляем rolling-фичи
    df_all = _add_rolling_features(df_base, bundle)

    # 3. Полный список фичей модели (должен быть в bundle)
    all_features = bundle['features']
    missing = set(all_features) - set(df_all.columns)
    if missing:
        raise ValueError(f"Отсутствуют фичи, необходимые модели: {missing}")

    # 4. Берём последнюю строку
    last_row = df_all.iloc[-1:]

    # 5. Выбираем нужные фичи в правильном порядке
    X = last_row[all_features].copy()

    # 6. Масштабируем (scaler ожидает DataFrame с такими же именами)
    X_scaled = pd.DataFrame(
        bundle['scaler'].transform(X),
        columns=all_features,
        index=X.index
    )

    # 7. Предсказание вероятности класса 1 (BUY)
    proba = bundle['model'].predict_proba(X_scaled)[0]
    # CatBoost возвращает вероятности для классов [0,1] в порядке возрастания?
    # В ноутбуке используется proba[:,1] – индекс 1 соответствует классу 1 (BUY)
    buy_prob = proba[1]

    # 8. Пороги из bundle
    thr_hi = bundle.get('threshold', 0.7)
    thr_lo = bundle.get('threshold_lo', 0.25)

    # 9. Применяем пороги
    if buy_prob >= thr_hi:
        return 1, buy_prob
    elif buy_prob <= thr_lo:
        return -1, 1 - buy_prob
    else:
        return 0, buy_prob