#!/usr/bin/env python3
import os

def create_dir(path):
    """Crea la carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creada carpeta: {path}")
    else:
        print(f"La carpeta ya existe: {path}")

def main():
    # Carpeta raíz del proyecto
    root_dir = "trading_bot_project"
    create_dir(root_dir)
    
    # Subcarpetas principales
    folders = [
        "data",             # Datos históricos y en tiempo real
        "models",           # Guardado de modelos (RL, LSTM, etc.)
        "logs",             # Archivos de registro y métricas de entrenamiento
        "notebooks",        # Jupyter Notebooks para análisis y pruebas
        "config",           # Archivos de configuración
        "src",              # Código fuente principal
    ]
    
    for folder in folders:
        create_dir(os.path.join(root_dir, folder))
    
    # Estructura dentro de src para organizar el código
    src_subfolders = [
        "data_ingestion",   # Scripts para conectar con IBKR y obtener datos
        "preprocessing",    # Limpieza y feature engineering
        "rl_agent",         # Implementación del agente de RL (PPO, etc.)
        "env",              # Definición del entorno (simulación, OpenAI Gym, etc.)
        "training",         # Código para el entrenamiento y backtesting
        "trading_bot",      # Bot de trading que interactúa con IBKR en paper mode
        "utils",            # Funciones y clases de utilidad
    ]
    
    for subfolder in src_subfolders:
        create_dir(os.path.join(root_dir, "src", subfolder))
    
    # Crear archivos README básicos en la raíz y en src
    readme_files = {
        root_dir: "# Trading Bot Project\n\nEste proyecto implementa un bot de trading basado en RL.",
        os.path.join(root_dir, "src"): "# Código Fuente\n\nAquí se encuentra el código principal del proyecto."
    }
    
    for folder, content in readme_files.items():
        readme_path = os.path.join(folder, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(content)
            print(f"Creado archivo: {readme_path}")
        else:
            print(f"El archivo ya existe: {readme_path}")

if __name__ == "__main__":
    main()
