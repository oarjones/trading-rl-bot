# src/rl_agent/ppo_agent.py

import tensorflow as tf
import numpy as np
import keras as ke

layers = ke.layers
Model = ke.Model
optimizers = ke.optimizers
losses = ke.losses

class Actor(Model):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.logits = layers.Dense(output_dim)  # Salida sin activación; se aplicará softmax al calcular probabilidades

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)  # Logits que se convertirán en probabilidades

class Critic(Model):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.value = layers.Dense(1)  # Valor escalar

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x)

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=4, mini_batch_size=64):
        # Asegúrate de que input_dim coincide con el vector de observación del entorno actualizado.
        self.actor = Actor(input_dim, action_dim)
        self.critic = Critic(input_dim)
        self.actor_optimizer = optimizers.Adam(lr)
        self.critic_optimizer = optimizers.Adam(lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        # Buffers para almacenar experiencias
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        
    def select_action(self, state):
        """
        Recibe un estado (como numpy array) y devuelve:
          - acción seleccionada (entero)
          - log_prob de la acción
        """
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        logits = self.actor(state_tensor)
        # Convertir logits a probabilidades usando softmax
        probs = tf.nn.softmax(logits)
        # Muestrear una acción
        action = tf.random.categorical(tf.math.log(probs), num_samples=1)
        action = tf.squeeze(action, axis=0).numpy()[0]
        # Calcular el logaritmo de la probabilidad de la acción seleccionada
        log_probs = tf.nn.log_softmax(logits)
        log_prob = log_probs[0, action].numpy()
        return action, log_prob

    def store_transition(self, state, action, log_prob, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def compute_advantages(self):
        """
        Calcula los retornos acumulados y las ventajas.
        Método simple sin GAE.
        """
        returns = []
        R = 0
        # Se calcula en orden inverso
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns, dtype=np.float32)
        
        states_tensor = tf.convert_to_tensor(np.array(self.states), dtype=tf.float32)
        values = tf.squeeze(self.critic(states_tensor)).numpy()
        advantages = returns - values
        return advantages, returns
    
    def update(self):
        advantages, returns = self.compute_advantages()
        
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)
        
        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)
        
        # Número de actualizaciones en cada epoch
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                batch_states = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
                
                # Actualizar actor
                with tf.GradientTape() as tape:
                    logits = self.actor(batch_states)
                    # Probabilidades y log_probabilidades
                    log_probs = tf.nn.log_softmax(logits)
                    # Extraer log_prob para la acción tomada
                    indices_batch = tf.stack([tf.range(tf.shape(batch_actions)[0]), batch_actions], axis=1)
                    new_log_probs = tf.gather_nd(log_probs, indices_batch)
                    
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Actualizar crítico
                with tf.GradientTape() as tape:
                    values = tf.squeeze(self.critic(batch_states), axis=1)
                    critic_loss = losses.mean_squared_error(batch_returns, values)
                    critic_loss = tf.reduce_mean(critic_loss)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        self.clear_buffer()
    
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()

# Ejemplo de uso para pruebas
if __name__ == "__main__":
    # Supongamos que la dimensión de la observación es 15
    input_dim = 15
    action_dim = 3
    agent = PPOAgent(input_dim, action_dim)
    
    # Ciclo de prueba: simulamos 200 pasos con estados aleatorios
    num_steps = 200
    dummy_state = np.random.randn(input_dim).astype(np.float32)
    
    for step in range(num_steps):
        action, log_prob = agent.select_action(dummy_state)
        reward = np.random.randn() * 0.1  # recompensa aleatoria para pruebas
        done = (step == num_steps - 1)
        next_state = np.random.randn(input_dim).astype(np.float32)
        agent.store_transition(dummy_state, action, log_prob, reward, done, next_state)
        dummy_state = next_state
    
    agent.update()
    print("Actualización completada con TensorFlow.")
