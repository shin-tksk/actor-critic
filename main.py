import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import utils

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
episode = 100
env = gym.make("CartPole-v1", render_mode='rgb_array')  # Create the environment
env.reset(seed=seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0　丸め誤差
frames = []

# model
num_inputs = 4
num_actions = 2
num_hidden = 256

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# training
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = [] #action log prob by step
critic_value_history = [] #critic by step
rewards_history = [] #reward by step

#while True:  # Run until solved
for epi in range(1,episode+1):

    state = env.reset()[0] #[cart-pos, cart-speed, pole-ang, pole-speed]
    episode_reward = 0
    with tf.GradientTape() as tape:
        #episode start
        for timestep in range(1, max_steps_per_episode):
            state = tf.convert_to_tensor(state) 
            state = tf.expand_dims(state, 0) #shape(1,4)
            action_probs, critic_value = model(state) #output[action(1,2)(left or right prob) ,critic(1,1)]
            critic_value_history.append(critic_value[0, 0])
            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs)) #left or right
            action_probs_history.append(tf.math.log(action_probs[0, action])) #selected dirction log prob
            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)[:4] #state(after apply action), reward(0 or 1), done(episode finished)
            rewards_history.append(reward)
            episode_reward += reward
            #frames.append(env.render())
            if done:
                break
        #episode finish
        returns = [] #cumulative reward
        discounted_sum = 0
        for r in rewards_history[::-1]: #where r is reward by step
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        actor_loss = sum(actor_losses)
        critic_loss = sum(critic_losses)
        loss_value = actor_loss + critic_loss
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    if epi % 1 == 0:
        print('loss : {}, reward : {} at episode {}'.format(loss_value, episode_reward, epi))

#utils.display_frames_as_gif(frames)
