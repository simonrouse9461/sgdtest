from smartgd.common.datasets import RomeDataset
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from smartgd.experiment.mixins import LoggingMixin

import os
from typing import Optional, Any
from abc import ABC, abstractmethod

import pytorch_lightning as L
import torch_geometric as pyg


class BaseLightningModule(L.LightningModule, LoggingMixin, ABC):

    def __init__(self, config: Any):
        super().__init__()

        if config:
            hparam_dict = self.generate_hyperparameters(config)
            self.save_hyperparameters(hparam_dict)

        self.dataset: Optional[pyg.data.Dataset] = None
        self.datamodule: Optional[L.LightningModule] = None

    def load_hyperparameters(self, hparam_dict: dict[str, Any]) -> None:
        self.hparams.clear()
        self.hparams.update(hparam_dict)

    @property
    def model_syncer(self) -> ModelSyncer:
        return ModelSyncer()

    @property
    def layout_syncer(self) -> LayoutSyncer:
        assert "dataset_name" in self.hparams, "'dataset_name' can not be found in hparams!"
        return LayoutSyncer.get_default_syncer(self.hparams.dataset_name)

    @abstractmethod
    def generate_hyperparameters(self, config: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_data(self) -> None:
        # TODO: dynamically load dataset by name
        dataset = RomeDataset()
        self.on_prepare_data(dataset)

    @abstractmethod
    def on_prepare_data(self, dataset: pyg.data.Dataset) -> None:
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        # TODO: dynamically load dataset by name
        self.dataset = RomeDataset()  # TODO: make sure it's not shuffled
        # TODO: create datamodule from within dataset
        self.datamodule = pyg.data.lightning.LightningDataset(
            train_dataset=self.dataset[:10000],
            val_dataset=self.dataset[11000:],
            test_dataset=self.dataset[10000:11000],
            batch_size=self.hparams.batch_size,
            num_workers=0#os.cpu_count()  # TODO: refactor s3_dataset_syncing to enable multiple workers
        )

    def train_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.test_dataloader()

