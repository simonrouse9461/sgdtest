from smartgd.common.data import GraphLayout, BaseTransformation, RescaleByStress
from smartgd.common.datasets import RomeDataset, BatchAppendColumn
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from smartgd.common.nn.criteria import (
    CompositeCritic,
    RGANCriterion,
    BaseAdverserialCriterion,
)
from smartgd.experiment.mixins import LoggingMixin

from typing import Optional, Any, Union

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg


class DeepGDLightningModule(L.LightningModule, LoggingMixin):
    def __init__(self, *,
                 dataset_name: str,
                 generator_name: Optional[str] = None,
                 generator_version: Optional[str] = None,
                 criteria: Union[str, dict[str, float]] = "stress_only",
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 lr_gamma: float = 0.998):
        super().__init__()

        # Models
        self.syncer: ModelSyncer = ModelSyncer()
        self.generator: Optional[nn.Module] = None

        # Data
        self.dataset: Optional[pyg.data.Dataset] = None
        self.datamodule: Optional[L.LightningModule] = None
        self.layout_manager: Optional[LayoutSyncer] = None

        # Functions
        self.adversarial_criterion: BaseAdverserialCriterion = RGANCriterion()
        self.canonicalize: BaseTransformation = RescaleByStress()
        self.append_column = BatchAppendColumn()

        # TODO: load from hparams
        if isinstance(criteria, str):
            self.critic = CompositeCritic.from_preset(criteria, batch_reduce=None)
        else:
            self.critic = CompositeCritic(criteria_weights=criteria, batch_reduce=None)

        # TODO: load hparams directly from hparams.yml for existing experiments
        self.save_hyperparameters(dict(
            dataset_name=dataset_name,
            generator=dict(
                meta=self.syncer.load_metadata(
                    name=generator_name,
                    version=generator_version
                ),
                args=self.syncer.load_arguments(
                    name=generator_name,
                    version=generator_version,
                    serialization=str
                )
            ),
            criteria_weights=self.critic.weights,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_gamma=lr_gamma
        ))

    def prepare_data(self):
        # TODO: dynamically load dataset by name
        self.dataset = RomeDataset()  # TODO: make sure it's not shuffled
        # TODO: create datamodule from within dataset
        self.datamodule = pyg.data.LightningDataset(
            train_dataset=self.dataset[:10000],
            val_dataset=self.dataset[11000:],
            test_dataset=self.dataset[10000:11000],
            batch_size=self.hparams.batch_size,
        )
        self.layout_manager = LayoutSyncer.get_default_syncer(self.dataset.name)

    def train_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> pyg.loader.DataLoader:
        return self.datamodule.test_dataloader()

    def setup(self, stage: str) -> None:
        if not self.generator:
            self.generator = self.syncer.load(
                name=self.hparams.generator["meta"]["model_name"],
                version=self.hparams.generator["meta"]["md5_digest"]
            )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.generator = self.syncer.load(
            name=checkpoint["hyper_parameters"]["generator"]["meta"]["model_name"],
            version=checkpoint["hyper_parameters"]["generator"]["meta"]["md5_digest"]
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        pass

    def forward(self, batch: pyg.data.Data):
        layout = GraphLayout.from_data(data=batch)
        layout = self.canonicalize(layout)
        layout = self.generator(layout)
        return layout

    def configure_callbacks(self) -> Union[L.Callback, list[L.Callback]]:
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

    def generator_optimizer(self) -> dict:
        return dict(
            optimizer=(optimizer := torch.optim.AdamW(
                params=self.generator.parameters(),
                lr=self.hparams.learning_rate
            )),
            lr_scheduler=dict(
                name="generator_optimizer",
                scheduler=torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=self.hparams.lr_gamma,
                    last_epoch=self.current_epoch - 1,
                    verbose=True
                ),
                frequency=1
            ),
        )

    def configure_optimizers(self) -> Any:
        return self.generator_optimizer()

    # def configure_optimizers(self) -> Any:
    #     return dict(
    #         optimizer=(optimizer := torch.optim.AdamW(
    #             params=self.parameters(),
    #             lr=1  # Set to 1 to allow LambdaLR to gain full control of lr
    #         )),
    #         lr_scheduler=dict(
    #             name="generator_optimizer",
    #             scheduler=torch.optim.lr_scheduler.LambdaLR(
    #                 optimizer=optimizer,
    #                 lr_lambda=lambda epoch: self.hparams.learning_rate,
    #                 last_epoch=self.current_epoch - 1,
    #                 verbose=True
    #             ),
    #             frequency=math.inf  # Only allow manual step() in the lr finder
    #         )
    #     )

    def training_step(self, batch: pyg.data.Batch, batch_idx: int) -> dict:
        fake_layout = self.canonicalize(self(batch))
        fake_score, fake_raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()

        loss = self.adversarial_criterion(encourage=fake_score, discourage=-fake_score)

        self.log_train(loss=loss.item(), score=fake_score.mean().item(),
                       **{k: v.mean().item() for k, v in fake_raw_scores.items()})
        return dict(loss=loss)

    def validation_step(self, batch: pyg.data.Batch, batch_idx: int):
        fake_layout = self.canonicalize(self(batch))
        score, raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()
        self.log_val(score=score.mean().item(),
                     **{k: v.mean().item() for k, v in raw_scores.items()})
