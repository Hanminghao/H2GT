from abc import ABC
import torch


class Evaluator(ABC):
    def __init__(self, config, verbose=True) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["datasets"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoint']
        self.config_gnn = config["GNN"]

        self.verbose = verbose

        # GPU configs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define checkpoint manager

