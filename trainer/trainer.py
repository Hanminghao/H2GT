from abc import ABC
from collections import OrderedDict
import torch


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["datasets"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoint']
        self.config_gnn = config["GNN"]
        self.config_eval = config['eval']

        # Read name from configs
        self.name = config['name']


        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Reading number of epochs
        self.n_epoch = self.config_train['num_epochs']

        # Number of workers and batch size
        self.num_workers = self.config_data["num_workers"]
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

    def train(self) -> None:
        raise NotImplementedError
