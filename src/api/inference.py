"""
Скрипт инференса модели
"""
import numpy as np
import joblib
from typing import List, Union

class TradingSignalPredictor:
    def __init__(self, model_path: str):
        """
        Инициализация предиктора
        
        Args:
            model_path: Путь к файлу модели
        """
        self.model = joblib.load(model_path)
        print(f"✅ Модель загружена из {model_path}")
    
    def predict(self, features: Union[List[List[float]], np.ndarray]) -> int:
        """
        Предсказание сигнала
        
        Args:
            features: Массив фичей (N шагов × M признаков)
            
        Returns:
            1 (BUY), -1 (SELL), или 0 (HOLD)
        """
        features_array = np.array(features)
        
        # Если передана последовательность, берем последний шаг
        if features_array.ndim == 2:
            features_array = features_array[-1, :]
        
        prediction = self.model.predict([features_array])[0]
        return int(prediction)

# Пример использования
if __name__ == "__main__":
    predictor = TradingSignalPredictor("models/baseline_v1.pkl")
    
    # Тестовый пример
    test_features = [
        [-0.5, 0.2, -0.1, 100.5, 12345.0],
        [-0.3, 0.1, -0.05, 100.6, 13456.0],
        [-0.1, 0.0, 0.0, 100.7, 14567.0]
    ]
    
    signal = predictor.predict(test_features)
    print(f"Предсказанный сигнал: {signal}")