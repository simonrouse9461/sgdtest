from smartgd.common.data import GraphLayout, BaseTransformation, RescaleByStress
from smartgd.common.datasets import RomeDataset, BatchAppendColumn
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from .criteria import (
    CompositeCritic,
    RGANCriterion,
    BaseAdverserialCriterion,
)
from .mixins import LoggingMixin

from typing import Optional, Any, Dict, List

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg


class SmartGDLightningModule(L.LightningModule, LoggingMixin):
    def __init__(self, *,
                 dataset_name: str,
                 generator_name: Optional[str] = None,
                 generator_version: Optional[str] = None,
                 discriminator_name: Optional[str] = None,
                 discriminator_version: Optional[str] = None,
                 criteria: Union[str, Dict[str, float]] = "stress_only",
                 alternating_mode: str = "step",
                 generator_frequency: Union[int, float] = 1,
                 discriminator_frequency: Union[int, float] = 1,
                 batch_size: int = 16,
                 learning_rate: float = 1e-3,
                 lr_gamma: float = 0.998):
        super().__init__()

        # Models
        self.syncer: ModelSyncer = ModelSyncer()
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None

        # Data
        self.dataset: Optional[pyg.data.Dataset] = None
        self.datamodule: Optional[L.LightningModule] = None
        self.layout_manager: Optional[LayoutSyncer] = None
        self.real_layout_store: Optional[Dict[str, np.ndarray]] = None

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
            discriminator=dict(
                meta=self.syncer.load_metadata(
                    name=discriminator_name,
                    version=discriminator_version
                ),
                args=self.syncer.load_arguments(
                    name=discriminator_name,
                    version=discriminator_version,
                    serialization=str
                )
            ),
            criteria_weights=self.critic.weights,
            alternating_mode=alternating_mode,
            generator_frequency=generator_frequency,
            discriminator_frequency=discriminator_frequency,
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
        self.real_layout_store = self.layout_manager.load(name="neato")

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
        if not self.discriminator:
            self.discriminator = self.syncer.load(
                name=self.hparams.discriminator["meta"]["model_name"],
                version=self.hparams.discriminator["meta"]["md5_digest"]
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self.generator = self.syncer.load(
            name=checkpoint["hparams"]["generator"]["meta"]["model_name"],
            version=checkpoint["hparams"]["generator"]["meta"]["md5_digest"]
        )
        self.discriminator = self.syncer.load(
            name=checkpoint["hparams"]["discriminator"]["meta"]["model_name"],
            version=checkpoint["hparams"]["discriminator"]["meta"]["md5_digest"]
        )
        self.real_layout_store = checkpoint["real_layout_store"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["real_layout_store"] = self.real_layout_store

    def forward(self, batch: pyg.data.Data):
        layout = GraphLayout.from_data(data=batch)
        layout = self.canonicalize(layout)
        layout = self.generator(layout)
        return layout

    def configure_callbacks(self) -> L.Callback | List[L.Callback]:
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

    def generator_optimizer(self, steps: int) -> dict:
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
            frequency=steps
        )

    def discriminator_optimizer(self, steps: int) -> dict:
        return dict(
            optimizer=(optimizer := torch.optim.AdamW(
                params=self.discriminator.parameters(),
                lr=self.hparams.learning_rate
            )),
            lr_scheduler=dict(
                name="discriminator_optimizer",
                scheduler=torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=self.hparams.lr_gamma,
                    last_epoch=self.current_epoch - 1,
                    verbose=True
                ),
                frequency=1
            ),
            frequency=steps
        )

    def configure_optimizers(self) -> Any:
        total_frequency = self.hparams.discriminator_frequency + self.hparams.generator_frequency
        # TODO: match case
        if self.hparams.alternating_mode == "step":
            discriminator_steps = self.hparams.discriminator_frequency
            generator_steps = self.hparams.generator_frequency
        elif self.hparams.alternating_mode == "epoch":
            steps_per_epoch = len(self.train_dataloader())
            total_steps = int(steps_per_epoch * total_frequency)
            discriminator_steps = int(steps_per_epoch * self.hparams.discriminator_frequency)
            generator_steps = total_steps - discriminator_steps
        else:
            assert False, f"Unknown alternating mode '{self.hparams.alternating_mode}'."
        return (
            self.discriminator_optimizer(discriminator_steps),
            self.generator_optimizer(generator_steps)
        )

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

    def training_step(self, batch: pyg.data.Batch, batch_idx: int, optimizer_idx: int) -> dict:
        fake_layout = self.canonicalize(self(batch))
        fake_pred = self.discriminator(fake_layout)
        fake_score, fake_raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()

        real_layout = self.canonicalize(GraphLayout.from_data(data=batch, kvstore=self.real_layout_store))
        real_pred = self.discriminator(real_layout)
        real_score = self.critic(real_layout)

        positive, negative = real_score > fake_score, real_score < fake_score
        good_pred = torch.cat([real_pred[positive], fake_pred[negative]])
        bad_pred = torch.cat([fake_pred[positive], real_pred[negative]])

        # TODO: match case
        if optimizer_idx == 0:  # discriminator
            loss = self.adversarial_criterion(encourage=good_pred, discourage=bad_pred)
        elif optimizer_idx == 1:  # generator
            loss = self.adversarial_criterion(encourage=fake_pred, discourage=real_pred)
        else:
            assert False, f"Unknown optimizer with index {optimizer_idx}."

        batch = self.append_column(batch=batch, tensor=fake_layout.pos, name="fake_pos")
        batch = self.append_column(batch=batch, tensor=negative, name="flagged")

        self.log_train(loss=loss.item(), score=fake_score.mean().item(),
                       **{k: v.mean().item() for k, v in fake_raw_scores.items()})
        return dict(loss=loss, batch=batch)

    def training_step_end(self, step_output: dict) -> torch.Tensor:
        batch = step_output["batch"]
        for data in batch.to_data_list():
            if data.flagged.item():
                self.real_layout_store[data.name] = data.fake_pos.detach().numpy()
        return step_output["loss"]

    def validation_step(self, batch: pyg.data.Data, batch_idx: int):
        fake_layout = self.canonicalize(self(batch))
        score, raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()
        self.log_val(score=score.mean().item(),
                     **{k: v.mean().item() for k, v in raw_scores.items()})
