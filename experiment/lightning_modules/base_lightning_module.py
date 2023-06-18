from smartgd.common.datasets import RomeDataset
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from smartgd.experiment.mixins import LoggingMixin

import os
import inspect
from typing import Optional, Any
from typing_extensions import Self
from abc import ABC, abstractmethod

import pytorch_lightning as L
import torch_geometric as pyg


class BaseLightningModule(L.LightningModule, LoggingMixin, ABC):

    dataset: Optional[pyg.data.Dataset]
    datamodule: Optional[L.LightningModule]

    training_step_outputs: list[dict[str, Any]]
    validation_step_outputs: list[dict[str, Any]]
    test_step_outputs: list[dict[str, Any]]

    def __init__(self, config: Optional[Any] = None):
        super().__init__()

        self._init_frame = inspect.currentframe()
        if config:
            hparam_dict = self.generate_hyperparameters(config)
            self.load_hyperparameters(hparam_dict)

        self.dataset = None
        self.datamodule = None

        # Result Buffers
        self.train_step_outputs = []  # TODO
        self.validation_step_outputs = []  # TODO
        self.test_step_outputs = []

    def load_hyperparameters(self, hparam_dict: dict[str, Any]) -> Self:
        hparam_dict = self.on_load_hyperparameters(hparam_dict)
        self.save_hyperparameters(hparam_dict, frame=self._init_frame)
        if self.loggers:
            for logger in self.loggers:
                logger.log_hyperparams(self.hparams)
        return self

    def on_load_hyperparameters(self, hparam_dict: dict[str, Any]) -> dict[str, Any]:
        return hparam_dict

    @property
    def model_syncer(self) -> ModelSyncer:
        return ModelSyncer()

    @property
    def layout_syncer(self) -> LayoutSyncer:
        assert "dataset_name" in self.hparams, "'dataset_name' can not be found in hparams!"
        return LayoutSyncer(dataset_name=self.hparams.dataset_name)

    @abstractmethod
    def generate_hyperparameters(self, config: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_data(self) -> None:
        # TODO: dynamically load dataset_name by name
        dataset = RomeDataset()
        self.on_prepare_data(dataset)

    def on_prepare_data(self, dataset: pyg.data.Dataset) -> None:
        pass

    def setup(self, stage: str) -> None:
        assert "train_slice" in self.hparams, "'train_slice' can not be found in hparams!"
        assert "val_slice" in self.hparams, "'val_slice' can not be found in hparams!"
        assert "test_slice" in self.hparams, "'test_slice' can not be found in hparams!"
        # TODO: dynamically load dataset_name by name
        self.dataset = RomeDataset()  # TODO: make sure it's not shuffled
        # TODO: create datamodule from within dataset_name
        self.datamodule = pyg.data.lightning.LightningDataset(
            train_dataset=self.dataset[slice(*self.hparams.train_slice)],
            val_dataset=self.dataset[slice(*self.hparams.val_slice)],
            test_dataset=self.dataset[slice(*self.hparams.test_slice)],
            batch_size=self.hparams.batch_size,
            num_workers=0#os.cpu_count()  # TODO: refactor s3_dataset_syncing to enable multiple workers
        )

    def train_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.test_dataloader()

    def on_train_epoch_start(self) -> None:
        self.train_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs.clear()
