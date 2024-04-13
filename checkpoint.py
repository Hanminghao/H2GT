import os
import json
from pathlib import Path
from typing import Optional, Dict, Generator, Tuple
import numpy as np
import sys
import torch


class CheckpointManager:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

        # Initial version of the checkpoint
        self.version = self.load_version()
        self.old_version = 0

        # Prepare checkpoint paths
        self.prepare()

        # Initialize training stats
        self.stats = {}

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def get_version_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "version.txt"

    def get_config_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "configs.json"

    def get_model_file(self, version: int, path: Optional[Path] = None) -> Path or Tuple:
        if path is None:
            path = self.path

        return path / f"model_v{version}.pt"

    def get_stats_file(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.path
        return path / "training_stats.json"

    def save_config(self, config: Dict) -> None:
        config_json = json.dumps(config, indent=4)
        with self.get_config_file().open("wt") as tf:
            tf.write(config_json)

    def load_config(self) -> str:
        try:
            with self.get_config_file().open("rt") as tf:
                return tf.read()
        except FileNotFoundError as err:
            raise err

    def append_stats(self, stats: Dict) -> None:
        stats_json = json.dumps(stats)
        with self.get_stats_file().open("at") as tf:
            tf.write(f"{stats_json}\n")

    def load_stats(self) -> Generator[str, None, None]:
        try:
            with self.get_stats_file().open("rt") as tf:
                for line in tf:
                    yield line
        except FileNotFoundError as err:
            raise err

    def save_model(
            self,
            state_dicts: Dict[str, torch.Tensor] or Tuple,
    ) -> None:
        """
        Save the embeddings into respective paths
        :param state_dicts: State dict of the model
        :return:
        """
        path = self.get_model_file(self.version)
        torch.save(state_dicts, path)

    def load_model(self) -> Dict[str, torch.Tensor]:
        path = self.get_model_file(self.version)
        state_dicts = torch.load(path)
        return state_dicts

    def save_version(self, version: int) -> None:
        with self.get_version_file().open("wt") as tf:
            tf.write(f"{version}\n")
            tf.flush()
            os.fsync(tf.fileno())

    def load_version(self) -> int:
        try:
            with self.get_version_file().open("rt") as tf:
                version_string = tf.read().strip()
        except FileNotFoundError:
            return 0
        else:
            if len(version_string) == 0:
                return 0
            else:
                return int(version_string)

    def write_new_version(
            self,
            config: Dict,
            state_dict: Optional[Dict[str, torch.Tensor] or Tuple],
            epoch_stats: Dict = None,
    ) -> None:
        """
        Write new version of checkpoint
        :param config: configurations
        :param state_dict: state dict of the model or tuple of state dicts
        :param epoch_stats: dictionary of stats
        :return:
        """
        if self.version == 0:
            self.save_config(config)

        # Update to new version
        self.old_version = self.version
        self.version = epoch_stats["Epoch"]
        self.save_version(self.version)

        # Save model state dict to folders
        self.save_model(state_dict)

        # Save training stats here
        # Format epoch stat
        for s, v in epoch_stats.items():
            if type(v) != int:
                epoch_stats[s] = round(v, 5)
        self.append_stats(epoch_stats)

    def remove_old_version(self) -> None:
        old_version = self.old_version

        # Remove older model
        path = self.get_model_file(old_version)

        try:
            path.unlink()
        except FileNotFoundError:
            pass


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation C-Index improved.
                            Default: 15
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation C-Index improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_cindex_max = 0

    def __call__(self, epoch, val_cindex, model, ckpt_name = 'checkpoint.pt'):

        score = val_cindex
        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            return ckpt_name
        elif score < self.best_score:
            self.counter += 1
            sys.stdout.write('\n')
            sys.stdout.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0
            return ckpt_name

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        '''Saves model when validation C-index increase.'''
        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.write(f'Validation C-Index increased ({self.val_cindex_max:.6f} --> {val_cindex:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_cindex_max = val_cindex
