from smartgd.common.data import (
    GraphStruct,
    GraphDrawingData
)
from smartgd.common.nn import (
    BaseTransformation,
    RescaleByStress,
    Compose,
    Center,
    NormalizeRotation,
    CompositeCritic,
    RGANCriterion,
    BaseAdverserialCriterion
)
from ..data_adaptors import DiscriminatorDataAdaptor, GeneratorDataAdaptor
from ..utils import RandomLayoutStore
from .base_lightning_module import BaseLightningModule

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any, Union

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg
from tqdm.auto import tqdm


class SmartGDLightningModule(BaseLightningModule):

    @dataclass
    class Config:
        dataset_name:               str = "Rome"
        generator_spec:             Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        discriminator_spec:         Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        criteria:                   Union[str, dict[str, float]] = "stress_only"
        static_transform:           Union[str, list[str], None] = None
        dynamic_transform:          Union[str, list[str], None] = None
        init_layout_method:         Optional[str] = "pmds"
        real_layout_candidates:     Union[str, list[str]] = field(default_factory=lambda: [
            "neato", "sfdp", "spring", "spectral", "kamada_kawai", "fa2", "pmds"
        ])
        alternating_mode:           str = "step"
        generator_frequency:        Union[int, float] = 1
        discriminator_frequency:    Union[int, float] = 1
        batch_size:                 int = 16
        learning_rate:              float = 1e-3
        lr_gamma:                   float = 0.998

    def __init__(self, config: Optional[Config]):
        super().__init__(config)

        # Models
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None

        # Data
        self.init_layout_store: Optional[dict[str, np.ndarray]] = None
        self.real_layout_store: Optional[dict[str, np.ndarray]] = None
        self.replacement_counter: Optional[dict[str, int]] = None

        # Functions
        self.critic: Optional[CompositeCritic] = None
        self.adversarial_criterion: BaseAdverserialCriterion = RGANCriterion()
        self.canonicalize: BaseTransformation = Compose(
            Center(),
            NormalizeRotation(),
            RescaleByStress()
        )

    def generate_hyperparameters(self, config: Config) -> dict[str, Any]:
        # TODO: load from hparams
        if isinstance(config.criteria, str):
            config.criteria = CompositeCritic.get_preset(config.criteria)
        if not isinstance(config.generator_spec, tuple):
            config.generator_spec = (config.generator_spec, None)
        if not isinstance(config.discriminator_spec, tuple):
            config.discriminator_spec = (config.discriminator_spec, None)
        if isinstance(config.real_layout_candidates, str):
            config.real_layout_candidates = [config.real_layout_candidates]
        if config.static_transform is None:
            config.static_transform = []
        if isinstance(config.static_transform, str):
            config.static_transform = [config.static_transform]
        if config.dynamic_transform is None:
            config.dynamic_transform = []
        if isinstance(config.dynamic_transform, str):
            config.dynamic_transform = [config.dynamic_transform]
        # TODO: load hparams directly from hparams.yml for existing experiments
        return dict(
            dataset_name=config.dataset_name,
            generator=dict(
                meta=self.model_syncer.load_metadata(
                    name=config.generator_spec[0],
                    version=config.generator_spec[1]
                ),
                args=self.model_syncer.load_arguments(
                    name=config.generator_spec[0],
                    version=config.generator_spec[1],
                    serialization=str
                )
            ),
            discriminator=dict(
                meta=self.model_syncer.load_metadata(
                    name=config.discriminator_spec[0],
                    version=config.discriminator_spec[1]
                ),
                args=self.model_syncer.load_arguments(
                    name=config.discriminator_spec[0],
                    version=config.discriminator_spec[1],
                    serialization=str
                )
            ),
            criteria_weights=config.criteria,
            static_transform=config.static_transform,
            dynamic_transform=config.dynamic_transform,
            init_layout_method=config.init_layout_method,
            real_layout_candidates=config.real_layout_candidates,
            alternating_mode=config.alternating_mode,
            generator_frequency=config.generator_frequency,
            discriminator_frequency=config.discriminator_frequency,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lr_gamma=config.lr_gamma
        )

    def on_prepare_data(self, dataset) -> None:
        if len(self.hparams.real_layout_candidates) <= 1:
            return
        layout_params = dict(
            candidates=self.hparams.real_layout_candidates,
            criteria=self.hparams.criteria_weights
        )
        if self.layout_syncer.exists(name="ranked", params=layout_params):
            return
        critic = CompositeCritic(
            criteria_weights=self.hparams.criteria_weights,
            batch_reduce=None
        )
        layout_stores = {
            layout: self.layout_syncer.load(name=layout)
            for layout in self.hparams.real_layout_candidates
        }
        real_layout_store = {}
        print("Generating real layouts...")
        for data in tqdm(dataset, desc="Generate Real"):
            data = data.post_transform(self.hparams.static_transform)
            scores = torch.cat([
                # TODO: merge data into batch first
                critic(data.struct(store, post_transform=self.hparams.dynamic_transform))
                for layout, store in layout_stores.items()
            ])
            best_layout = list(layout_stores)[scores.argmin()]
            real_layout_store[data.name] = layout_stores[best_layout][data.name]
        if not self.layout_syncer.exists(name="ranked", params=layout_params):
            print("Uploading real layouts...")
            self.layout_syncer.save(real_layout_store, name="ranked", params=layout_params)

    def load_generator(self, generator_config: dict[str, Any]):
        self.generator = GeneratorDataAdaptor(self.model_syncer.load(
            name=generator_config["meta"]["model_name"],
            version=generator_config["meta"]["md5_digest"],
        ))

    def load_discriminator(self, discriminator_config: dict[str, Any]):
        self.discriminator = DiscriminatorDataAdaptor(self.model_syncer.load(
            name=discriminator_config["meta"]["model_name"],
            version=discriminator_config["meta"]["md5_digest"],
        ))

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if not self.generator:
            self.load_generator(self.hparams.generator)
        if not self.discriminator:
            self.load_discriminator(self.hparams.discriminator)
        self.critic = CompositeCritic(
            criteria_weights=self.hparams.criteria_weights,
            batch_reduce=None
        )
        if len(self.hparams.real_layout_candidates) == 0:
            raise NotImplementedError  # TODO: use random layout
        if len(self.hparams.real_layout_candidates) == 1:
            assert self.layout_syncer.exists(name=self.hparams.real_layout_candidates[0]), "Layout not found!"
            self.real_layout_store = self.layout_syncer.load(name=self.hparams.real_layout_candidates[0])
        else:
            layout_params = dict(
                candidates=self.hparams.real_layout_candidates,
                criteria=self.hparams.criteria_weights
            )
            assert self.layout_syncer.exists(name="ranked", params=layout_params), "Layout not found!"
            self.real_layout_store = self.layout_syncer.load(name="ranked", params=layout_params)
        if self.hparams.init_layout_method:
            self.init_layout_store = self.layout_syncer.load(name=self.hparams.init_layout_method)
        else:
            template_layout_store = self.layout_syncer.load(name="random")
            self.init_layout_store = RandomLayoutStore(template=template_layout_store)
        self.replacement_counter = defaultdict(int)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.load_generator(checkpoint["hyper_parameters"]["generator"])
        self.load_discriminator(checkpoint["hyper_parameters"]["discriminator"])
        self.real_layout_store = checkpoint["real_layout_store"]
        self.replacement_counter = checkpoint["replacement_counter"]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint["real_layout_store"] = self.real_layout_store
        checkpoint["replacement_counter"] = self.replacement_counter

    def forward(self, batch: pyg.data.Data):
        layout = batch.struct(self.init_layout_store)
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

    def training_step(self, batch: GraphDrawingData, batch_idx: int, optimizer_idx: int) -> dict:
        batch = batch.post_transform(self.hparams.static_transform)
        fake_layout = self(batch)
        fake_layout = self.canonicalize(fake_layout)
        fake_layout = batch.struct(fake_layout, post_transform=self.hparams.dynamic_transform)
        fake_pred = self.discriminator(fake_layout)
        fake_score, fake_raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()

        real_layout = batch.struct(self.real_layout_store)
        real_layout = self.canonicalize(real_layout)
        real_layout = batch.struct(real_layout, post_transform=self.hparams.dynamic_transform)
        real_pred = self.discriminator(real_layout)
        real_score = self.critic(real_layout)

        positive, negative = real_score < fake_score, real_score > fake_score
        good_pred = torch.cat([real_pred[positive], fake_pred[negative]])
        bad_pred = torch.cat([fake_pred[positive], real_pred[negative]])

        discriminator_loss = self.adversarial_criterion(encourage=good_pred, discourage=bad_pred)
        generator_loss = self.adversarial_criterion(encourage=fake_pred, discourage=real_pred)

        # TODO: match case
        if optimizer_idx == 0:  # discriminator
            loss = discriminator_loss
        elif optimizer_idx == 1:  # generator
            loss = generator_loss
        else:
            assert False, f"Unknown optimizer with index {optimizer_idx}."

        batch = batch.append(tensor=fake_layout.pos, name="fake_pos")
        batch = batch.append(tensor=negative, name="flagged")

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

    def on_training_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        self.log_epoch_end(
            total_replacements=sum(self.replacement_counter.values()),
            total_unique_replacements=len(self.replacement_counter)
        )

    def validation_step(self, batch: pyg.data.Batch, batch_idx: int):
        batch = batch.post_transform(self.hparams.static_transform)
        # TODO: all metrics need to be evaluated
        # TODO: evaluate SPC
        fake_layout = self.canonicalize(self(batch))
        fake_layout = batch.struct(fake_layout, post_transform=self.hparams.dynamic_transform)
        score, raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()
        self.log_val_step(
            score=score.mean().item(),
            **{k: v.mean().item() for k, v in raw_scores.items()}
        )
