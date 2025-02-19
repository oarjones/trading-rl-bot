#!/usr/bin/env python3
"""
Script para preprocesar datos históricos y generar indicadores técnicos,
incluyendo la normalización de los campos OHLC y de todos los indicadores calculados.
Se añaden nuevas columnas con el sufijo _norm.
"""

import pandas as pd
import os
import ta
import numpy as np
import ta.momentum
import ta.trend
import ta.volatility

def normalize_series(series):
    """Normaliza una serie usando min-max scaling, devolviendo valores en [0,1]."""
    if series.max() == series.min():
        return series - series.min()  # Si es constante, devuelve 0.
    return (series - series.min()) / (series.max() - series.min())

def preprocess_data_for_symbol(symbol):
    input_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = os.path.join(input_dir, f"{symbol}.csv")
    df = pd.read_csv(input_file)
    
    # Verificar que existan las columnas esenciales de OHLC
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en los datos de {symbol}.")
    
    # Eliminar filas con valores faltantes
    df.dropna(inplace=True)
    
    # Normalizar los valores OHLC y agregar columnas con sufijo _norm
    for col in ['open', 'high', 'low', 'close']:
        df[f"{col}_norm"] = normalize_series(df[col])
    
    # CALCULAR INDICADORES (usando el precio normalizado cuando tenga sentido)
    # 1. SMA_50: para capturar una tendencia a mediano plazo (50 horas ≈ 2 días de trading)
    df['SMA_50'] = ta.trend.sma_indicator(df['close_norm'], window=50)
    # 2. RSI_14: RSI con ventana 14 (14 horas)
    df['RSI_14'] = ta.momentum.rsi(df['close_norm'], window=14)
    # 3. MACD_diff: usando los parámetros tradicionales (12,26,9) sobre close_norm
    df['MACD_diff'] = ta.trend.macd_diff(df['close_norm'])
    # 4. ATR_14: se calcula sobre los valores originales, ya que mide volatilidad absoluta
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    # 5. Bollinger Bands: sobre close_norm, ventana 20 y desviación 2
    bollinger = ta.volatility.BollingerBands(df['close_norm'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    # 6. ADX_14: indicador de fuerza de la tendencia, usando datos originales (o se puede usar close_norm)
    df['ADX_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # CALCULAR NUEVO INDICADOR DE TENDENCIA (TrendStrength)
    # Normalizar RSI_14 a [-1,1] usando (RSI - 50)/50
    df['RSI_component'] = (df['RSI_14'] - 50) / 50
    # Normalizar MACD_diff con tanh para acotarlo a [-1,1]
    df['MACD_component'] = np.tanh(df['MACD_diff'])
    # Combinar ambos con pesos iguales (0.5 cada uno)
    w_rsi = 0.5
    w_macd = 0.5
    df['TrendComposite'] = w_rsi * df['RSI_component'] + w_macd * df['MACD_component']
    # Mapear a [0,1]: 1 = fuerte alcista, 0 = fuerte bajista, 0.5 = neutral
    df['TrendStrength'] = (df['TrendComposite'] + 1) / 2
    
    # NORMALIZAR LOS INDICADORES CALCULADOS (añadiendo sufijo _norm)
    # Para RSI_14, que varía de 0 a 100, se normaliza dividiendo entre 100
    df['RSI_14_norm'] = df['RSI_14'] / 100.0
    # Para SMA_50
    df['SMA_50_norm'] = normalize_series(df['SMA_50'])
    # Para MACD_diff
    df['MACD_diff_norm'] = normalize_series(df['MACD_diff'])
    # Para ATR_14
    df['ATR_14_norm'] = normalize_series(df['ATR_14'])
    # Para Bollinger Bands
    df['BB_High_norm'] = normalize_series(df['BB_High'])
    df['BB_Low_norm'] = normalize_series(df['BB_Low'])
    # Para RSI_component y MACD_component, que están en [-1,1], mapear a [0,1]
    df['RSI_component_norm'] = (df['RSI_component'] + 1) / 2
    df['MACD_component_norm'] = (df['MACD_component'] + 1) / 2
    # Para TrendComposite
    df['TrendComposite_norm'] = normalize_series(df['TrendComposite'])
    # TrendStrength ya está en [0,1]; duplicamos la columna con sufijo si se desea
    df['TrendStrength_norm'] = df['TrendStrength']
    # Para ADX_14, dado que sus valores pueden variar, se normaliza con min-max
    df['ADX_14_norm'] = normalize_series(df['ADX_14'])


    # Ejemplo de limpieza: eliminar filas con valores faltantes
    df.dropna(inplace=True)
    
    # (Opcional) Ajustar el formato de la fecha si existe columna 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Guardar el DataFrame preprocesado
    output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"Datos preprocesados para {symbol} guardados en {output_file}")

if __name__ == "__main__":
    
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "BAC", "GS", "C",
        "WMT", "TGT", "COST",
        "JNJ", "PFE", "MRK",
        "BA", "GE", "CAT",
        "XOM", "CVX",
        "T", "VZ",
        "DIS", "NFLX", "IBM"
    ]


    for symbol in symbols:
        try:
            preprocess_data_for_symbol(symbol)
        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
