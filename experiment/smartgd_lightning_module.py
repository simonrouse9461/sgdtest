from smartgd.common.data import (
    GraphLayout,
    BaseTransformation,
    RescaleByStress,
    Compose,
    Center,
    NormalizeRotation
)
from smartgd.common.datasets import RomeDataset, BatchAppendColumn
from smartgd.common.syncing import LayoutSyncer, ModelSyncer
from smartgd.common.nn.criteria import (
    CompositeCritic,
    RGANCriterion,
    BaseAdverserialCriterion,
)
from .base_lightning_module import BaseLightningModule

from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg


class SmartGDLightningModule(BaseLightningModule):

    @dataclass
    class Config:
        dataset_name: str
        generator_spec: Union[Optional[str], Tuple[Optional[str], Optional[str]]] = None
        discriminator_spec: Union[Optional[str], Tuple[Optional[str], Optional[str]]] = None
        criteria: Union[str, Dict[str, float]] = "stress_only"
        alternating_mode: str = "step"
        generator_frequency: Union[int, float] = 1
        discriminator_frequency: Union[int, float] = 1
        batch_size: int = 16
        learning_rate: float = 1e-3
        lr_gamma: float = 0.998

    def __init__(self, config: Optional[Config]):
        super().__init__(config)

        # Models
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None

        # Data
        self.layout_manager: Optional[LayoutSyncer] = None
        self.real_layout_store: Optional[Dict[str, np.ndarray]] = None
        self.replacement_counter: Optional[Dict[str, int]] = None

        # Functions
        self.critic: Optional[CompositeCritic] = None
        self.adversarial_criterion: BaseAdverserialCriterion = RGANCriterion()
        self.canonicalize: BaseTransformation = Compose(
            Center(),
            NormalizeRotation(),
            RescaleByStress()
        )
        self.append_column: BatchAppendColumn = BatchAppendColumn()

    def generate_hyperparameters(self, config: Config) -> Dict[str, Any]:
        # TODO: load from hparams
        if isinstance(config.criteria, str):
            config.criteria = CompositeCritic.get_preset(config.criteria)
        if not isinstance(config.generator_spec, Tuple):
            config.generator_spec = (config.generator_spec, None)
        if not isinstance(config.discriminator_spec, Tuple):
            config.discriminator_spec = (config.discriminator_spec, None)
        # TODO: load hparams directly from hparams.yml for existing experiments
        return dict(
            dataset_name=config.dataset_name,
            generator=dict(
                meta=self.syncer.load_metadata(
                    name=config.generator_spec[0],
                    version=config.generator_spec[1]
                ),
                args=self.syncer.load_arguments(
                    name=config.generator_spec[0],
                    version=config.generator_spec[1],
                    serialization=str
                )
            ),
            discriminator=dict(
                meta=self.syncer.load_metadata(
                    name=config.discriminator_spec[0],
                    version=config.discriminator_spec[1]
                ),
                args=self.syncer.load_arguments(
                    name=config.discriminator_spec[0],
                    version=config.discriminator_spec[1],
                    serialization=str
                )
            ),
            criteria_weights=config.criteria,
            alternating_mode=config.alternating_mode,
            generator_frequency=config.generator_frequency,
            discriminator_frequency=config.discriminator_frequency,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lr_gamma=config.lr_gamma
        )

    def prepare_data(self):
        super().prepare_data()
        self.layout_manager = LayoutSyncer.get_default_syncer(self.dataset.name)
        # TODO: load by hparams
        self.real_layout_store = self.layout_manager.load(name="neato")
        self.replacement_counter = {k: 0 for k in self.real_layout_store}

    def setup(self, stage: str) -> None:
        if not self.critic:
            self.critic = CompositeCritic(
                criteria_weights=self.hparams.criteria_weights,
                batch_reduce=None
            )
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
            name=checkpoint["hyper_parameters"]["generator"]["meta"]["model_name"],
            version=checkpoint["hyper_parameters"]["generator"]["meta"]["md5_digest"]
        )
        self.discriminator = self.syncer.load(
            name=checkpoint["hyper_parameters"]["discriminator"]["meta"]["model_name"],
            version=checkpoint["hyper_parameters"]["discriminator"]["meta"]["md5_digest"]
        )
        self.real_layout_store = checkpoint["real_layout_store"]
        self.real_layout_store = checkpoint["replacement_counter"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["real_layout_store"] = self.real_layout_store
        checkpoint["replacement_counter"] = self.replacement_counter

    def forward(self, batch: pyg.data.Data):
        layout = GraphLayout.from_data(data=batch)
        layout = self.canonicalize(layout)
        layout = self.generator(layout)
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

        positive, negative = real_score < fake_score, real_score > fake_score
        good_pred = torch.cat([real_pred[positive], fake_pred[negative]])
        bad_pred = torch.cat([fake_pred[positive], real_pred[negative]])

        # discriminator_loss = self.adversarial_criterion(encourage=good_pred, discourage=bad_pred)
        # discriminator_loss = self.adversarial_criterion(encourage=real_pred, discourage=fake_pred)
        discriminator_loss = ((fake_pred + fake_score.log()).square() + (real_pred + real_score.log()).square()).mean()
        generator_loss = self.adversarial_criterion(encourage=fake_pred, discourage=real_pred)

        # TODO: match case
        if optimizer_idx == 0:  # discriminator
            loss = discriminator_loss
        elif optimizer_idx == 1:  # generator
            loss = generator_loss
        else:
            assert False, f"Unknown optimizer with index {optimizer_idx}."

        batch = self.append_column(batch=batch, tensor=fake_layout.pos, name="fake_pos")
        batch = self.append_column(batch=batch, tensor=negative, name="flagged")

        self.log_train_step(
            discriminator_loss=discriminator_loss.item(),
            generator_loss=generator_loss.item(),
            score=fake_score.mean().item(),
            **{k: v.mean().item() for k, v in fake_raw_scores.items()}
        )
        return dict(loss=loss, batch=batch)

    def training_step_end(self, step_output: dict) -> torch.Tensor:
        batch = step_output["batch"]
        replacements = 0
        initial_replacements = 0
        for data in batch.to_data_list():
            if data.flagged.item():
                self.real_layout_store[data.name] = data.fake_pos.detach().cpu().numpy()
                self.replacement_counter[data.name] += 1
                if self.replacement_counter[data.name] == 1:
                    initial_replacements += 1
                replacements += 1
        self.log_train_step_sum_on_epoch_end(
            replacements=replacements,
            initial_replacements=initial_replacements
        )
        return step_output["loss"]

    def training_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        self.log_epoch_end(
            total_replacements=sum(self.replacement_counter.values()),
            total_unique_replacements=sum(map(bool, self.replacement_counter.values()))
        )

    def validation_step(self, batch: pyg.data.Batch, batch_idx: int):
        fake_layout = self.canonicalize(self(batch))
        score, raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()
        self.log_val_step(
            score=score.mean().item(),
            **{k: v.mean().item() for k, v in raw_scores.items()}
        )
