#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polling worker для интеграции с платформой.
Опрашивает эндпоинт /api/ml/ds/feature-windows, получает окна признаков,
вызывает модель через src.api.inference.predict и отправляет сигналы обратно.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Добавляем корень проекта в sys.path, чтобы импортировать src.api.inference
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.api.inference import predict
from integration.config import (
    API_BASE,
    POLL_INTERVAL,
    REQUEST_TIMEOUT,
    SIGNAL_SOURCE,
    CONFIDENCE_THRESHOLD,
)


def get_feature_windows():
    """Запрашивает окна признаков у платформы (только READY)."""
    try:
        response = requests.get(
            f"{API_BASE}/api/ml/ds/feature-windows?readyOnly=true",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Worker] ❌ Ошибка получения фичей: {e}")
        return None


def send_signal(payload):
    """Отправляет сигнал в платформу."""
    # Убедимся, что source установлен из конфига
    payload["source"] = SIGNAL_SOURCE
    try:
        response = requests.post(
            f"{API_BASE}/api/signals/ingest",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 400:
            print(f"[Worker] ❌ Ошибка 400 (неверные данные). Payload: {payload}")
            return
        response.raise_for_status()
        print(f"[Worker] ✅ Сигнал {payload['signal']} отправлен")
    except requests.exceptions.RequestException as e:
        print(f"[Worker] ⚠️ Ошибка отправки: {e}")


def run_iteration():
    """Одна итерация опроса и обработки."""
    print("\n[Worker] 🔄 Запрос окон фичей...")
    data = get_feature_windows()
    if not data:
        return

    feature_columns = data.get("featureColumns", [])
    for item in data.get("items", []):
        if item.get("state") != "READY":
            print(f"[Worker] ⏩ Пропуск {item.get('symbol')} (state={item.get('state')})")
            continue

        symbol = item["symbol"]
        features = item["features"]
        window_end = item.get("windowEndTimestamp")

        # Создаём DataFrame из полученной матрицы
        df = pd.DataFrame(features, columns=feature_columns)
        df["symbol"] = symbol

        # Генерируем временные метки, если их нет в колонках
        # Предполагаем, что данные идут с интервалом 1 минута
        df["timestamp"] = [
            window_end - (len(df) - 1 - i) * 60000 for i in range(len(df))
        ]

        # Переименовываем 'close' в 'close_price', если нужно
        if "close" in df.columns and "close_price" not in df.columns:
            df.rename(columns={"close": "close_price"}, inplace=True)

        # Вызываем модель
        try:
            signal, confidence = predict(df)
        except Exception as e:
            print(f"[Worker] ❌ Ошибка при предсказании для {symbol}: {e}")
            continue

        # Применяем порог уверенности и фильтруем HOLD
        if signal == 0 or confidence < CONFIDENCE_THRESHOLD:
            print(
                f"[Worker] ⏸️ HOLD для {symbol} "
                f"(signal={signal}, conf={confidence:.2f}, threshold={CONFIDENCE_THRESHOLD})"
            )
            continue

        # Формируем полезную нагрузку
        signal_str = "BUY" if signal == 1 else "SELL"
        # Берём цену закрытия из последней строки (после возможного переименования)
        close_price_col = "close_price" if "close_price" in df.columns else "close"
        close_price = df.iloc[-1][close_price_col]

        payload = {
            "symbol": symbol,
            "timestamp": window_end,
            "signal": signal_str,
            "price": float(close_price),
            "rating": round(confidence, 4),
            # source будет добавлен в send_signal из конфига
        }
        send_signal(payload)


def main():
    print(f"🚀 Worker запущен. Опрос {API_BASE} каждые {POLL_INTERVAL} сек.")
    print(f"   Порог уверенности: {CONFIDENCE_THRESHOLD}, источник: {SIGNAL_SOURCE}")
    while True:
        run_iteration()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()