import argparse
from importlib import import_module

import numpy as np
import torch

from train_eval import train, test
from utils import build_dataset, build_iterator

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',type=str,required=True,help='choose a model:bert')
args = parser.parse_args()

if __name__ == '__main__':

    dataset = 'toutiao'

    if args.model == 'bert':

        model_name = 'bert'
        x = import_module(model_name)
        config = x.Config(dataset)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True  #保证每次结果一样 cudnn底层也采用了随机数算法

        print('Loading data for Bert Model...')

        train_data,dev_data,test_data = build_dataset(config)

        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter)
        test(config, model, test_iter)



