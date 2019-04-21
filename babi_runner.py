import glob
import os
import random
import sys
import pickle
import gzip

import argparse
import numpy as np

from config import BabiConfig, BabiConfigJoint
from train_test import train, train_linear_start, test
from util import parse_babi_task, build_model
from demo.qa import MemN2N

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)  # for reproducing




def run_test(data_dir, task_id, memn2n):
    print("Test for task %d ..." % task_id)
    test_files = None
    if type(data_dir) is tuple:
        test_files = glob.glob('%s/qa%d_*_valid.txt' % (data_dir[1], task_id))
    else:
        test_files = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))

    test_story, test_questions, test_qstory = parse_babi_task(test_files, memn2n.general_config.dictionary, False)

    """
    reversed_dict = None
    memory = None
    model = None
    loss = None
    general_config = None

    with gzip.open(model_file, "rb") as f:
        self.reversed_dict, self.memory, self.model, self.loss, self.general_config = pickle.load(f)
    """
    test(test_story, test_questions, test_qstory, memn2n.memory, memn2n.model, memn2n.loss, memn2n.general_config)

def run_all_tests(data_dir, memn2n):
    print("Training and testing for all tasks ...")
    for t in range(1, 21):
        run_test(data_dir, t,  memn2n)

def run_task(data_dir, task_id):
    """
    Train and test for each task
    """
    print("Train and test for task %d ..." % task_id)

    # Parse data
    train_files = glob.glob('%s/qa%d_*_train.txt' % (data_dir, task_id))
    test_files  = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))

    dictionary = {"nil": 0}
    train_story, train_questions, train_qstory = parse_babi_task(train_files, dictionary, False)
    test_story, test_questions, test_qstory    = parse_babi_task(test_files, dictionary, False)

    general_config = BabiConfig(train_story, train_questions, dictionary)

    memory, model, loss = build_model(general_config)

    if general_config.linear_start:
        train_linear_start(train_story, train_questions, train_qstory, memory, model, loss, general_config)
    else:
        train(train_story, train_questions, train_qstory, memory, model, loss, general_config)

    test(test_story, test_questions, test_qstory, memory, model, loss, general_config)


def run_all_tasks(data_dir):
    """
    Train and test for all tasks
    """
    print("Training and testing for all tasks ...")
    for t in range(20):
        run_task(data_dir, task_id=t + 1)


def run_joint_tasks(data_dir):
    """
    Train and test for all tasks but the trained model is built using training data from all tasks.
    """
    print("Jointly train and test for all tasks ...")
    tasks = range(20)

    # Parse training data
    train_data_path = []
    for t in tasks:
        train_data_path += glob.glob('%s/qa%d_*_train.txt' % (data_dir, t + 1))

    dictionary = {"nil": 0}
    train_story, train_questions, train_qstory = parse_babi_task(train_data_path, dictionary, False)

    # Parse test data for each task so that the dictionary covers all words before training
    for t in tasks:
        test_data_path = glob.glob('%s/qa%d_*_test.txt' % (data_dir, t + 1))
        parse_babi_task(test_data_path, dictionary, False) # ignore output for now

    general_config = BabiConfigJoint(train_story, train_questions, dictionary)
    memory, model, loss = build_model(general_config)

    if general_config.linear_start:
        train_linear_start(train_story, train_questions, train_qstory, memory, model, loss, general_config)
    else:
        train(train_story, train_questions, train_qstory, memory, model, loss, general_config)

    # Test on each task
    for t in tasks:
        print("Testing for task %d ..." % (t + 1))
        test_data_path = glob.glob('%s/qa%d_*_test.txt' % (data_dir, t + 1))
        dc = len(dictionary)
        test_story, test_questions, test_qstory = parse_babi_task(test_data_path, dictionary, False)
        assert dc == len(dictionary)  # make sure that the dictionary already covers all words

        test(test_story, test_questions, test_qstory, memory, model, loss, general_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", default="data/tasks_1-20_v1-2/en",
                        help="path to dataset directory (default: %(default)s)")

    parser.add_argument("-m", "--model-file", default="trained_model/memn2n_model.pklz",
                        help="model file (default: %(default)s)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--task", default="1", type=int,
                       help="train and test for a single task (default: %(default)s)")
    group.add_argument("-a", "--all-tasks", action="store_true",
                       help="train and test for all tasks (one by one) (default: %(default)s)")
    group.add_argument("-j", "--joint-tasks", action="store_true",
                       help="train and test for all tasks (all together) (default: %(default)s)")

    group.add_argument("-s", "--test", default="1", type=int,
                       help="test for a single task (default: %(default)s)")
    group.add_argument("-k", "--all-tests", action="store_true",
                       help="test for all tasks (one by one) (default: %(default)s)")

    parser.add_argument("-d2", "--data-dir2", default=None,
                        help="path to directory containing a training and testing directory)")

    args = parser.parse_args()

    # Check if data is available
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print("The data directory '%s' does not exist. Please download it first." % data_dir)
        sys.exit(1)

    if args.data_dir2 is not None:
        if not os.path.exists(args.data_dir2):
            print("The data directory '%s' does not exist." % args.data_dir)
            sys.exit(1)
        else:
            train_path = os.path.join(args.data_dir2, 'train')
            if not os.path.exists(train_path):
                print("'%s' does not exist." % train_path)
                sys.exit(1)
            test_path = os.path.join(args.data_dir2, 'test')
            if not os.path.exists(test_path):
                print("'%s' does not exist." % test_path)
                sys.exit(1)
            args.data_dir = train_path, test_path

    if type(args.data_dir) is tuple:
        print("Using data from {} and {}".format(args.data_dir[0], args.data_dir[1]))
    else:
        print("Using data from %s" % args.data_dir)

    if args.test or args.all_tests:
        m = MemN2N(args.data_dir, args.model_file)
        m.load_model()
        if args.all_tests:
            run_all_tests(data_dir, m)
        else:
            run_test(data_dir, args.task, m)
    else:
        if args.all_tasks:
            run_all_tasks(data_dir)
        elif args.joint_tasks:
            run_joint_tasks(data_dir)
        else:
            run_task(data_dir, task_id=args.task)
