# src/training/train_agent.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from src.env.trading_env import AdvancedTradingEnv
from src.rl_agent.ppo_agent import PPOAgent

# Parámetros de entrenamiento
NUM_EPISODES = 200         # Número de episodios de entrenamiento
UPDATE_INTERVAL = 1        # Actualizar el agente al final de cada episodio (podrías ajustar esto)
PRINT_INTERVAL = 10        # Imprimir métricas cada 10 episodios

# Cargar datos preprocesados (asegúrate de que el CSV procesado contenga todos los indicadores normalizados)
# Por ejemplo, usamos el archivo para AAPL; en un escenario real podrías entrenar con múltiples activos
data_path = os.path.join("data", "processed", "AAPL_processed.csv")
data = pd.read_csv(data_path)

# Inicializar el entorno. Suponemos que en datos de 1 hour, un día de trading equivale a 7 barras.
env = AdvancedTradingEnv(data,
                         initial_cash=3000.0,
                         trading_fee=0.001,
                         stop_loss=0.02,
                         take_profit=0.04,
                         episode_length=7,           # 1 día de trading
                         lstm_feedback_enabled=False)  # Cambia a True si tienes feedback del LSTM

# Determinar la dimensión de la observación a partir del espacio definido en el entorno
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Inicializar el agente PPO (asegúrate de que el agente implementa los métodos necesarios)
agent = PPOAgent(input_dim=obs_dim,
                 action_dim=action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 update_epochs=4,
                 mini_batch_size=32)  # Puedes ajustar el tamaño de mini-batch

# Listas para almacenar métricas
episode_rewards = []
nav_history = []

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        # Seleccionar acción y obtener log_prob (según PPOAgent)
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, log_prob, reward, done, next_state)
        state = next_state
        episode_reward += reward

    # Actualizar el agente al final del episodio (o cada UPDATE_INTERVAL episodios)
    agent.update()

    # Guardar métricas del episodio
    episode_rewards.append(episode_reward)
    nav_history.append(info.get('net_asset_value', env.cash))

    if episode % PRINT_INTERVAL == 0:
        avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
        print(f"Episodio {episode}/{NUM_EPISODES} - Promedio recompensa (últimos {PRINT_INTERVAL}): {avg_reward:.2f} - NAV: {nav_history[-1]:.2f}")

# Al final del entrenamiento, graficamos la evolución del NAV y la recompensa acumulada
plt.figure(figsize=(12, 5))
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
