from smartgd.common.data import GraphLayout, BaseTransformation, RescaleByStress
from smartgd.common.datasets import RomeDataset, BatchAppendColumn
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from .base_lightning_module import BaseLightningModule

from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg


class DistillationLightningModule(BaseLightningModule):

    @dataclass
    class Config:
        dataset_name: str
        teacher_spec: Union[Optional[str], Tuple[Optional[str], Optional[str]]] = None
        student_spec: Union[Optional[str], Tuple[Optional[str], Optional[str]]] = None
        batch_size: int = 16
        learning_rate: float = 1e-3
        lr_gamma: float = 0.998

    def __init__(self, config: Optional[Config]):
        super().__init__(config)

        # Models
        self.teacher: Optional[nn.Module] = None
        self.student: Optional[nn.Module] = None

        # Functions
        self.canonicalize: BaseTransformation = RescaleByStress()
        self.append_column = BatchAppendColumn()

    def generate_hyperparameters(self, config: Config) -> Dict[str, Any]:
        # TODO: load hparams directly from hparams.yml for existing experiments
        if not isinstance(config.teacher_spec, Tuple):
            config.teacher_spec = (config.teacher_spec, None)
        if not isinstance(config.student_spec, Tuple):
            config.student_spec = (config.student_spec, None)
        return dict(
            dataset_name=config.dataset_name,
            teacher=dict(
                meta=self.syncer.load_metadata(
                    name=config.teacher_spec[0],
                    version=config.teacher_spec[1]
                ),
                args=self.syncer.load_arguments(
                    name=config.teacher_spec[0],
                    version=config.teacher_spec[1],
                    serialization=str
                )
            ),
            student=dict(
                meta=self.syncer.load_metadata(
                    name=config.student_spec[0],
                    version=config.student_spec[1]
                ),
                args=self.syncer.load_arguments(
                    name=config.student_spec[0],
                    version=config.student_spec[1],
                    serialization=str
                )
            ),
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lr_gamma=config.lr_gamma
        )

    def setup(self, stage: str) -> None:
        if not self.teacher:
            self.teacher = self.syncer.load(
                name=self.hparams.teacher["meta"]["model_name"],
                version=self.hparams.teacher["meta"]["md5_digest"]
            )
        if not self.student:
            self.student = self.syncer.load(
                name=self.hparams.student["meta"]["model_name"],
                version=self.hparams.student["meta"]["md5_digest"]
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self.teacher = self.syncer.load(
            name=checkpoint["hyper_parameters"]["teacher"]["meta"]["model_name"],
            version=checkpoint["hyper_parameters"]["teacher"]["meta"]["md5_digest"]
        )
        self.student = self.syncer.load(
            name=checkpoint["hyper_parameters"]["student"]["meta"]["model_name"],
            version=checkpoint["hyper_parameters"]["student"]["meta"]["md5_digest"]
        )

    def forward(self, batch: pyg.data.Data):
        layout = GraphLayout.from_data(data=batch)
        layout = self.canonicalize(layout)
        layout = self.student(layout)
        return layout

    def configure_callbacks(self) -> Union[L.Callback, List[L.Callback]]:
        return [
            # PeriodicLRFinder(
            #     interval=1,
            #     num_training_steps=50,
            #     # early_stop_threshold=None
            # ),
            L.callbacks.LearningRateMonitor(
                logging_interval="epoch",
                log_momentum=True
            )
        ]

    def configure_optimizers(self) -> Any:
        return dict(
            optimizer=(optimizer := torch.optim.AdamW(
                params=self.student.parameters(),
                lr=self.hparams.learning_rate
            )),
            lr_scheduler=dict(
                name="student_optimizer",
                scheduler=torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=self.hparams.lr_gamma,
                    last_epoch=self.current_epoch - 1,
                    verbose=True
                ),
                frequency=1
            ),
        )

    def training_step(self, batch: pyg.data.Batch, batch_idx: int) -> dict:
        layout = self.canonicalize(GraphLayout.from_data(data=batch))
        pred = self.student(layout)
        gt = self.teacher(layout)

        loss = (pred - gt).square().mean()

        self.log_train(loss=loss.item())
        return dict(loss=loss)

    def validation_step(self, batch: pyg.data.Batch, batch_idx: int):
        layout = self.canonicalize(GraphLayout.from_data(data=batch))
        pred = self.student(layout)
        gt = self.teacher(layout)

        loss = (pred - gt).square().sum()

        self.log_val(loss=loss.item())
