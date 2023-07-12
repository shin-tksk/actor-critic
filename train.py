import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
from model import ActorCriticModel
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default=None, type=str)
args = parser.parse_args()

opt_name = args.optimizer

parser = argparse.ArgumentParser()
learning_rate = 0.001
episode = 10000
num_step = 1000
loader_path = None

if opt_name == 'adam':
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

elif opt_name == 'radam':
    opt = RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

save_path = 'model/' + opt_name
train_log_dir = 'logs/' + opt_name + '/'



ac_model = ActorCriticModel(num_hidden=128, num_inputs=4, num_actions=2, loader_path=loader_path, gamma=0.99, learning_rate=learning_rate)
ac_model.compile(optimizer=opt)
# define tensorboard writer
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

best_reward = 0
best_epi = 0
reward_list = []

for epi in range(episode):
    loss, reward = ac_model.train_episode(num_step)
    reward_list.append(reward)

    #if reward > best_reward:
    #    best_reward = reward
    #    best_epi = epi

    if epi % 10 == 0:
        print('loss : {}, reward : {} at episode {}'.format(loss, reward, epi))
        
    #Tensor board
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=epi)
        tf.summary.scalar('reward', reward, step=epi)

    if epi >= 10:
        run_rewards = sum(reward_list[epi-9:epi+1]) / 10
        #print(run_rewards)
        if run_rewards > 300:
            ac_model.save_weight(save_path)
            break

print('Training is finish at {} episode., reward : {}'.format(epi, run_rewards))
