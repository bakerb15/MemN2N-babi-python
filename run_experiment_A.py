
from demo.qa import MemN2N
import os


# Default location for data is location on the HPCC
#path_to_babi_data = '/mnt/research/CSE842/FinalProject/Benchmarks/bAbI'
path_to_babi_data = '/home/bbaker/bAbI'

path_to_model_output_dir = './ExperimentA'

'''
Train, test, and save model for three different learning rates each with/without linear start 
and 1,2,3 memory modules (hops).

'''


starting_learning_rate = [.005, .1, .01]
linear_start = [True, False]
hops = [1, 2, 3]

config_swtiches = [None, None, None]

for lr in starting_learning_rate:
    config_swtiches[0] = lr
    for opt in linear_start:
        config_swtiches[1] = opt
        for num in hops:
            config_swtiches[2] = num
            model_file_name = 'lr{}_linearstart{}_hops{}'.format(*config_swtiches)
            write_model_to = os.path.join(path_to_model_output_dir, model_file_name)
            message = 'Training and testing model for {}'.format(model_file_name)
            print(message)
            model = MemN2N(path_to_babi_data, write_model_to, config_swtiches)
            model.train_and_test()
            print('\n\n')

