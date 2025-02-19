#!/usr/bin/env python3
"""
Script para obtener datos históricos desde IBKR y guardarlos en CSVs.

Requisitos:
- Tener instalado ib_insync.
- Tener IBKR TWS o IB Gateway en modo paper mode (o real, según convenga).
- Definir la lista de símbolos, el período a recuperar y el tamaño de barra.
"""

from ib_insync import IB, Stock, util
import pandas as pd
import os
import datetime

def fetch_and_save_data(symbols, duration="1 Y", barSize="4 hours", whatToShow="MIDPOINT", useRTH=True):
    # Conectarse a IBKR (ajusta IP, puerto y clientId según tu configuración)
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=1)
    
    # Asegúrate de que existe la carpeta para datos crudos
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    # Usaremos la hora actual como punto final (se podría parametrizar)
    endDateTime = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
    
    for symbol in symbols:
        print(f"Recuperando datos para {symbol}...")
        contract = Stock(symbol, 'SMART', 'USD')
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=endDateTime,
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=1
            )
            # Convertir a DataFrame usando la utilidad de ib_insync
            df = util.df(bars)
            # Guarda en CSV
            output_file = os.path.join(output_dir, f"{symbol}.csv")
            df.to_csv(output_file, index=False)
            print(f"Datos guardados en {output_file}")
        except Exception as e:
            print(f"Error al recuperar datos para {symbol}: {e}")
    
    ib.disconnect()

if __name__ == "__main__":
    # Lista de símbolos a recuperar
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

    

    # Define el periodo a recuperar, por ejemplo "1 Y" para 1 año; también puedes usar "2 Y", etc.
    fetch_and_save_data(symbols, duration="2 Y", barSize="1 hour")
