from smartgd.common.datasets import RomeDataset
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from .mixins import LoggingMixin

import os
from typing import Optional, Any
from abc import ABC, abstractmethod

import pytorch_lightning as L
import torch_geometric as pyg


class BaseLightningModule(L.LightningModule, LoggingMixin, ABC):

    def __init__(self, config: Any):
        super().__init__()

        self.syncer: ModelSyncer = ModelSyncer()

        if config:
            hyperparameters = self.generate_hyperparameters(config)
            self.save_hyperparameters(hyperparameters)

        self.dataset: Optional[pyg.data.Dataset] = None
        self.datamodule: Optional[L.LightningModule] = None

    def load_hyperparameters(self, hparam_dict: dict[str, Any]):
        self.hparams.clear()
        self.hparams.update(hparam_dict)

    @abstractmethod
    def generate_hyperparameters(self, config: Any) -> dict[str, Any]:
        return NotImplemented

    def prepare_data(self):
        # TODO: dynamically load dataset by name
        self.dataset = RomeDataset()  # TODO: make sure it's not shuffled
        # TODO: create datamodule from within dataset
        self.datamodule = pyg.data.LightningDataset(
            train_dataset=self.dataset[:10000],
            val_dataset=self.dataset[11000:],
            test_dataset=self.dataset[10000:11000],
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count()
        )

    def train_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.test_dataloader()

