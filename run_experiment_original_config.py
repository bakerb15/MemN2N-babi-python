
from demo.qa import MemN2N
import os
import random


# Default location for data is location on the HPCC
#path_to_babi_data = '/mnt/research/CSE842/FinalProject/Benchmarks/bAbI'
path_to_babi_data = '/home/bbaker/bAbI'

path_to_model_output_dir = './ExperimentA'

'''
Train, test, and save model from original repo configuration

'''


for i in range(10):
    seed = random.randint(1, 10000000)
    model_file_name = 'default_config_model{}_seed'.format(i)
    write_model_to = os.path.join(path_to_model_output_dir, model_file_name)
    message = 'Training and testing model for {}'.format(model_file_name)
    print(message)
    model = MemN2N(path_to_babi_data, write_model_to)
    model.train_and_test(seed)
    print('\n\n')

