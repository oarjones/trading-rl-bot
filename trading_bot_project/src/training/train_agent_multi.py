# src/training/train_agent_multi.py

import sys
import os

# Agrega la carpeta raíz del proyecto al sys.path:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from src.env.trading_env import AdvancedTradingEnv
from src.rl_agent.ppo_agent import PPOAgent
import random

# Lista de símbolos a utilizar
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

# Directorios donde se encuentran los datos preprocesados
processed_dir = os.path.join("data", "processed")

# Función para cargar y dividir los datos de un símbolo en entrenamiento y validación (80/20)
def load_and_split_symbol(symbol, split_ratio=0.8):
    filepath = os.path.join(processed_dir, f"{symbol}_processed.csv")
    df = pd.read_csv(filepath)
    # Se asume que los datos están ordenados cronológicamente.
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df

# Cargar los datos de todos los símbolos
train_data = {}
test_data = {}
for sym in symbols:
    try:
        tr, te = load_and_split_symbol(sym)
        train_data[sym] = tr
        test_data[sym] = te
    except Exception as e:
        print(f"Error al cargar datos para {sym}: {e}")

# Configuración del entrenamiento
NUM_EPISODES = 2000        # Número total de episodios de entrenamiento
PRINT_INTERVAL = 50        # Mostrar métricas cada 50 episodios
EPISODE_LENGTH = 7         # Cada episodio es 1 día de trading (7 pasos, 1 hora cada uno)

# Suponemos que el vector de observación se define en el entorno (ya actualizado para 1 hora)
# Se creará una instancia del entorno para cada episodio a partir del conjunto de entrenamiento de un símbolo seleccionado.
# Para ello, se asume que el entorno AdvancedTradingEnv toma un DataFrame que representa la serie de datos para el símbolo.

# Para determinar el input_dim, se crea una instancia temporal del entorno con AAPL (por ejemplo) y se consulta su observation_space.
temp_df = train_data["AAPL"]
temp_env = AdvancedTradingEnv(temp_df, episode_length=EPISODE_LENGTH, lstm_feedback_enabled=False)
obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.n
del temp_env

# Inicializar el agente PPO
agent = PPOAgent(input_dim=obs_dim,
                 action_dim=action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 update_epochs=4,
                 mini_batch_size=32)

# Listas para almacenar métricas de entrenamiento
episode_rewards = []
nav_history = []
symbol_history = []

# Función para obtener un episodio aleatorio de entrenamiento de un símbolo dado
def sample_episode(df, episode_length):
    # Seleccionar aleatoriamente un punto de inicio que permita un episodio completo
    if len(df) <= episode_length:
        raise ValueError("El DataFrame es demasiado corto para un episodio.")
    start_idx = np.random.randint(0, len(df) - episode_length)
    episode_df = df.iloc[start_idx:start_idx + episode_length].reset_index(drop=True)
    return episode_df

# Entrenamiento multi-símbolo
for episode in range(1, NUM_EPISODES + 1):
    # Seleccionar aleatoriamente un símbolo para este episodio (de entre los que tengan datos)
    sym = random.choice(list(train_data.keys()))
    symbol_history.append(sym)
    df_train = train_data[sym]
    # Muestrear un episodio aleatorio del DataFrame de entrenamiento
    episode_df = sample_episode(df_train, EPISODE_LENGTH)
    
    # Crear una instancia del entorno con el episodio seleccionado
    env = AdvancedTradingEnv(episode_df, 
                             initial_cash=3000.0,
                             trading_fee=0.001,
                             stop_loss=0.02,
                             take_profit=0.04,
                             episode_length=EPISODE_LENGTH,
                             lstm_feedback_enabled=False)
    
    state = env.reset()
    done = False
    ep_reward = 0.0
    
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, log_prob, reward, done, next_state)
        state = next_state
        ep_reward += reward
    
    # Actualizar el agente con la experiencia acumulada del episodio
    agent.update()
    
    episode_rewards.append(ep_reward)
    nav_history.append(info.get('net_asset_value', env.cash))
    
    if episode % PRINT_INTERVAL == 0:
        avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
        print(f"Episodio {episode}/{NUM_EPISODES} - Símbolo: {sym} - Promedio recompensa (últimos {PRINT_INTERVAL}): {avg_reward:.2f} - NAV: {nav_history[-1]:.2f}")

# Graficar métricas de entrenamiento
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(nav_history)
plt.title("Evolución del Patrimonio Neto (NAV)")
plt.xlabel("Episodio")
plt.ylabel("NAV")

plt.subplot(1, 2, 2)
plt.plot(episode_rewards)
plt.title("Recompensa acumulada por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.tight_layout()
plt.show()
