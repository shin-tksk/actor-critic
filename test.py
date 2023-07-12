import gym
from model import ActorCriticModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default=None, type=str)
args = parser.parse_args()
opt_name = args.optimizer

num_step = 200
loader_path = 'model/' + opt_name

ac_model = ActorCriticModel(num_hidden=128, num_inputs=4, num_actions=2, loader_path=loader_path, gamma=0.99)
utils.set_seed(16)

count = 0
for i in range(100):
    count += ac_model.test(num_step)

print(count,'game clear')