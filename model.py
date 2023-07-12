import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
import utils
import json




class ActorCriticModel(tf.keras.Model):
    def __init__(self, num_hidden=128, num_inputs=4, num_actions=2, loader_path=None, env=None, gamma=0.99, learning_rate=0.001):
        super(ActorCriticModel, self).__init__()

        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self.num_hidden = num_hidden
            self.num_inputs = num_inputs
            self.num_actions = num_actions
            self.gamma = gamma

        self.env = gym.make("CartPole-v1", render_mode='rgb_array')
        self.env.reset(seed=42)
        self.eps = np.finfo(np.float32).eps.item() # 丸め誤差

        self.common1 = tf.keras.layers.Dense(self.num_hidden, activation="relu")
        self.common2 = tf.keras.layers.Dense(self.num_hidden//2, activation="relu")
        self.action = tf.keras.layers.Dense(self.num_actions, activation="softmax")
        self.critic = tf.keras.layers.Dense(1)

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, state):
        state = tf.convert_to_tensor(state) 
        state = tf.expand_dims(state, 0) #shape(1,4)
        out1 = self.common1(state)
        out2 = self.common2(out1)
        action_probs = self.action(out2)
        critic_value = self.critic(out2)
        return action_probs, critic_value

    def train_episode(self, num_step):
        action_probs_history = [] #action log prob by step
        critic_value_history = [] #critic by step
        rewards_history = [] #reward by step
        state = self.env.reset()[0]
        episode_reward = 0

        with tf.GradientTape() as tape:
            for step in range(num_step):
                state, reward, done, log_action_probs, critic_value = self.train_step(state)
                critic_value_history.append(critic_value[0, 0])
                action_probs_history.append(log_action_probs) #selected dirction log prob
                rewards_history.append(reward)
                episode_reward += reward
                if done:
                    break

            returns = [] #cumulative reward
            discounted_sum = 0
            for r in rewards_history[::-1]: #where r is reward by step
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
            returns = returns.tolist()

            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                a_loss = -log_prob * diff 
                c_loss = utils.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0), delta=1.0)
                actor_losses.append(a_loss)  # actor loss
                critic_losses.append(c_loss)
            actor_loss = sum(actor_losses) / len(actor_losses)
            critic_loss = sum(critic_losses) / len(critic_losses)

            # Backpropagation
            self.loss_value = actor_loss + critic_loss
            gradients = tape.gradient(self.loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self.loss_value[0], episode_reward

    def train_step(self, state):
        action_probs, critic_value = self.call(state)
        action = np.random.choice(self.num_actions, p=np.squeeze(action_probs)) 
        log_action_probs = tf.math.log(action_probs[0, action])
        state, reward, done, _ = self.env.step(action)[:4] 
        return state, reward, done, log_action_probs, critic_value

    def test(self, num_step):
        state = self.env.reset()[0]
        for step in range(num_step):
            action_probs, _ = self.call(state)
            action = np.random.choice(self.num_actions, p=np.squeeze(action_probs)) 
            state, reward, done, _ = self.env.step(action)[:4] 
            if done:
                #print(state)
                print('game over, {} steps'.format(step))
                return 0
        print('game clear!')
        return 1

    def get_config(self):
        config = {}
        config['num_hidden'] = self.num_hidden
        config['num_inputs'] = self.num_inputs
        config['num_actions'] = self.num_actions
        config['gamma'] = self.gamma
        return config

    def __load_config(self, config):
        print(config)
        self.num_hidden = config['num_hidden']
        self.num_inputs = config['num_inputs']
        self.num_actions = config['num_actions']
        self.gamma = config['gamma']

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def save_weight(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'
        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")
        config = {}
        config['num_hidden'] = self.num_hidden
        config['num_inputs'] = self.num_inputs
        config['num_actions'] = self.num_actions
        config['gamma'] = self.gamma
        return config

    
