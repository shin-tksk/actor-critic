import gym
import numpy as np
import tensorflow as tf
import os
from model import ActorCriticModel
from tensorflow.keras.optimizers import Adam

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

learning_rate = 0.001
episode = 100
save_path = 'model'
loader_path = None

opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ac_model = ActorCriticModel(num_hidden=256, num_inputs=4, num_actions=2, loader_path=loader_path, gamma=0.99, learning_rate=learning_rate)
ac_model.compile(optimizer=opt)


for epi in range(1,episode+1):
    loss, reward = ac_model.train_episode()
    if epi % 1 == 0:
        print('loss : {}, reward : {} at episode {}'.format(loss, reward, epi))

ac_model.save_weight('model')