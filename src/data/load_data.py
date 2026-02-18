"""
Модуль для загрузки данных
"""
import pandas as pd
from pathlib import Path

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Загружает сырые данные из CSV
    
    Args:
        file_path: Путь к CSV файлу
        
    Returns:
        DataFrame с данными
    """
    df = pd.read_csv(file_path)
    print(f"✅ Загружено {len(df)} строк из {file_path}")
    return df

def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Сохраняет обработанные данные
    
    Args:
        df: DataFrame для сохранения
        file_path: Путь для сохранения
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"✅ Сохранено {len(df)} строк в {file_path}")