from globals import *

import yaml
import argparse
import json
import statistics
import random
import numpy as np
import torch
import os

from utils import ordered_yaml
from trainer import GNNTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('--seed', type=int, help='random seed of the run', default=42)
parser.add_argument('--gpu_list', type=str, help='Which Gpus to use', default='1')
parser.add_argument('--foldlist', nargs='+', type=int, help='Which fold to use', default=[1,2,3,4,5])
parser.add_argument('--info', type=str)
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

opt_path = args.config
gpu_list = args.gpu_list

default_config_path = "BRCA/HoverNet_KimiaNet.yml"
if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path
else:
    opt_path= CONFIG_DIR  / opt_path
# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mode = args.mode

def main():
    torch.autograd.set_detect_anomaly(True)
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")
    if args.info:
        config['desc'] = args.info
    val_dict = {}
    test_dict = {}
    fold_list = args.foldlist
    print('Training/ Testing in {nums} fold'.format(nums=len(fold_list)))
    for fold in fold_list:
        print('**'*100)
        print('Trianing/ Testing in fold {fold}'.format(fold=fold))
        if mode == "train":
            if config["train_type"] == "gnn":
                print(config['desc'])
                path = os.path.join(config['checkpoint']['path'], config['desc'], str(fold), 'model_v{}.pt'.format(config['train']['num_epochs']))
                if os.path.exists(path):
                    print('This model has been trained')
                    continue
                trainer = GNNTrainer(config, fold=fold)
            else:
                raise NotImplementedError("This type of model is not implemented")
            val_CIndex = trainer.train()
            val_dict[fold] = val_CIndex

        elif mode == 'test':
            if config["train_type"] == "gnn":
                trainer = GNNTrainer(config, fold=fold)
            else:
                raise NotImplementedError("This type of model is not implemented")
            folder_path = os.path.join(config['pkl_path'], str(fold))
            for filename in os.listdir(folder_path):
                if filename.endswith('.pt'):
                    file_path = os.path.join(folder_path, filename)
                    break  # 找到一个.pt文件后终止遍历
            trainer.best_model_path = file_path
            test_cindex = trainer.test()
            test_dict[fold] = test_cindex
    if mode == 'train':
        save_path = os.path.join(config['checkpoint']['path'],config['desc'])
        epoch_num = config['train']['num_epochs']
        for idx in range(10, epoch_num+1, 10):
            fold_result = []
            for fold in range(1,6):
                path = os.path.join(save_path, str(fold), 'training_stats.json')
                with open(path, 'r') as file:
                    line_number = 0
                    for line in file:
                        data = json.loads(line)
                        line_number += 1
                        if line_number == idx:
                            fold_result.append(data['Validation C-Index'])
            print('**'*20)
            print(fold_result)
            mean = sum(fold_result) / len(fold_result)
            std_dev = statistics.stdev(fold_result)
            print('epoch:{epoch}, mean:{mean}, std_dev:{std_dev}'.format(epoch=idx, mean=mean, std_dev=std_dev))
    else:
        print('**'*20)
        print(test_dict)
        mean = sum(test_dict.values()) / len(test_dict)
        std_dev = statistics.stdev(test_dict.values())
        print('mean:{mean}, std_dev:{std_dev}'.format(mean=mean, std_dev=std_dev))




if __name__ == "__main__":
    main()


