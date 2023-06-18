from smartgd.common.data import (
    GraphStruct,
    GraphDrawingData,
)
from smartgd.common.nn import (
    BaseTransformation,
    RescaleByStress,
    Standardization,
    Compose,
    Center,
    NormalizeRotation,
    CompositeMetric,
    RGANCriterion,
    BaseAdverserialCriterion,
    SPC
)
from ..data_adaptors import DiscriminatorDataAdaptor, GeneratorDataAdaptor
from ..layout_stores import RandomLayoutStore
from .base_lightning_module import BaseLightningModule

from functools import reduce
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any, Union, TypeVar

import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as L
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm


LayoutDict = dict[str, np.ndarray]


class SmartGDLightningModule(BaseLightningModule):

    @dataclass
    class Config:
        dataset_name:               str = "Rome"
        generator_spec:             Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        discriminator_spec:         Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        criteria:                   Union[str, dict[str, float]] = "stress_only"
        optional_data_fields:       list[str] = field(default_factory=list)
        eval_optional_data_fields:  list[str] = field(default_factory=list)
        train_slice:                tuple = (None, 10000)
        val_slice:                  tuple = (11000, None)
        test_slice:                 tuple = (10000, 11000)
        init_layout_method:         Optional[str] = "pmds"
        real_layout_candidates:     Union[str, list[str]] = field(default_factory=lambda: [
                                        "neato", "sfdp", "spring", "spectral", "kamada_kawai", "fa2", "pmds"
                                    ])
        learn_from_critic:          bool = True
        replace_by_critic:          bool = True
        alternating_mode:           str = "step"
        generator_frequency:        Union[int, float] = 1
        discriminator_frequency:    Union[int, float] = 1
        batch_size:                 int = 16
        learning_rate:              float = 1e-3
        lr_gamma:                   float = 0.998

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)

        # Models
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None

        # Data
        # TODO: this is problematic
        # GraphDrawingData.set_optional_fields(self.hparams.optional_data_fields)
        self.init_layout_store: Optional[LayoutDict] = None
        self.real_layout_store: Optional[LayoutDict] = None
        self.fake_layout_store: LayoutDict = {}
        self.replacement_counter: Optional[dict[str, int]] = None

        # Functions
        self.critic: Optional[CompositeMetric] = None
        self.eval_metric: CompositeMetric = CompositeMetric.from_preset("human_preference")
        self.adversarial_criterion: BaseAdverserialCriterion = RGANCriterion()
        self.canonicalize: BaseTransformation = Compose(
            Center(),
            NormalizeRotation(),
            # Standardization(),
            RescaleByStress(),
        )
        self.spc: SPC = SPC(batch_reduce=None)

    def generate_hyperparameters(self, config: Config) -> dict[str, Any]:
        # TODO: load from hparams
        if isinstance(config.criteria, str):
            config.criteria = CompositeMetric.get_preset(config.criteria)
        if not isinstance(config.generator_spec, tuple):
            config.generator_spec = (config.generator_spec, None)
        if not isinstance(config.discriminator_spec, tuple):
            config.discriminator_spec = (config.discriminator_spec, None)
        if isinstance(config.real_layout_candidates, str):
            config.real_layout_candidates = [config.real_layout_candidates]
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
            optional_data_fields=config.optional_data_fields,
            eval_optional_data_fields=config.eval_optional_data_fields,
            train_slice=config.train_slice,
            val_slice=config.val_slice,
            test_slice=config.test_slice,
            init_layout_method=config.init_layout_method,
            real_layout_candidates=config.real_layout_candidates,
            learn_from_critic=config.learn_from_critic,
            replace_by_critic=config.replace_by_critic,
            alternating_mode=config.alternating_mode,
            generator_frequency=config.generator_frequency,
            discriminator_frequency=config.discriminator_frequency,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lr_gamma=config.lr_gamma
        )

    def on_load_hyperparameters(self, hparam_dict: dict[str, Any]) -> dict[str, Any]:
        GraphDrawingData.set_optional_fields(hparam_dict['optional_data_fields'])  # TODO: context manager
        return hparam_dict

    def on_prepare_data(self, dataset) -> None:
        if len(self.hparams.real_layout_candidates) <= 1:
            return
        layout_params = self.generate_real_layout_params()
        # TODO: this is a hack
        if len(self.layout_syncer.load(**layout_params)) == len(dataset):
            return
        layout_stores = {
            layout: self.layout_syncer.load_layout_dict(name=layout)
            for layout in self.hparams.real_layout_candidates
        }
        real_layout_store = {}
        real_layout_metadata = {}
        print("Generating real layouts...")
        for data in tqdm(dataset, desc="Generate Real"):
            batch = Batch.from_data_list([
                data.clone().load_pos_dict(pos_dict=store)
                for store in layout_stores.values()
            ])
            score = self.evaluate_layout(
                batch=batch,
                metric=CompositeMetric(
                    criteria_weights=self.hparams.criteria_weights,
                    batch_reduce=None
                ),
                discriminate=False
            )[1]
            best_layout = list(layout_stores)[score.argmin()]
            real_layout_store[data.name] = layout_stores[best_layout][data.name]
            real_layout_metadata[data.name] = dict(method=best_layout)
        # TODO: hack
        if len(self.layout_syncer.load(**layout_params)) < len(dataset):
            print("Uploading real layouts...")
            self.layout_syncer.batch_update(
                layout_dict=real_layout_store,
                metadata_dict=real_layout_metadata,
                **layout_params
            )

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

    def generate_real_layout_params(self):
        if len(self.hparams.real_layout_candidates) == 0:
            return dict(name="random")
        elif len(self.hparams.real_layout_candidates) == 1:
            return dict(name=self.hparams.real_layout_candidates[0])
        else:
            return dict(name="ranked", params=dict(
                candidates=self.hparams.real_layout_candidates,
                criteria=self.hparams.criteria_weights
            ))

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if not self.generator:
            self.load_generator(self.hparams.generator)
        if not self.discriminator:
            self.load_discriminator(self.hparams.discriminator)
        self.critic = CompositeMetric(
            criteria_weights=self.hparams.criteria_weights,
            batch_reduce=None
        )
        real_layout_params = self.generate_real_layout_params()
        # TODO: hack
        assert len(self.layout_syncer.load(**real_layout_params)) == len(self.dataset), "Layout not found!"
        self.real_layout_store = self.layout_syncer.load_layout_dict(**real_layout_params)
        if self.hparams.init_layout_method:
            self.init_layout_store = self.layout_syncer.load_layout_dict(name=self.hparams.init_layout_method)
        else:
            self.init_layout_store = RandomLayoutStore(template=self.layout_syncer.load_layout_dict(name="random"))
        self.replacement_counter = defaultdict(int)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.load_generator(checkpoint["hyper_parameters"]["generator"])
        self.load_discriminator(checkpoint["hyper_parameters"]["discriminator"])
        self.real_layout_store = checkpoint["real_layout_store"]
        self.replacement_counter = checkpoint["replacement_counter"]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint["real_layout_store"] = self.real_layout_store
        checkpoint["replacement_counter"] = self.replacement_counter

    def on_train_start(self) -> None:
        GraphDrawingData.set_optional_fields(self.hparams.optional_data_fields)

    def on_validation_start(self) -> None:
        GraphDrawingData.set_optional_fields(
            self.hparams.optional_data_fields + self.hparams.eval_optional_data_fields
        )

    def on_test_start(self) -> None:
        GraphDrawingData.set_optional_fields(self.hparams.eval_optional_data_fields)

    def forward(self, batch: GraphDrawingData):
        layout = batch.make_struct(self.init_layout_store)
        layout = batch.transform_struct(self.canonicalize, layout)
        layout = batch.transform_struct(self.generator, layout)
        self.fake_layout_store.update(batch.pos_dict())
        return layout

    def configure_callbacks(self) -> Union[L.Callback, list[L.Callback]]:
        return [
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

    def evaluate_layout(self,
                        batch: GraphDrawingData,
                        layout: Optional[GraphStruct] = None,
                        metric: Optional[CompositeMetric] = None,
                        discriminate: bool = True):
        layout = layout or batch.make_struct()
        metric = metric or self.critic
        layout = batch.transform_struct(self.canonicalize, layout)
        score, raw_scores = metric(layout), metric.get_raw_scores()
        pred = self.discriminator(layout) if discriminate else None
        return pred, score, raw_scores

    def training_step(self, batch: GraphDrawingData, batch_idx: int, optimizer_idx: int) -> dict:
        fake_pred, fake_score, fake_raw_scores = self.evaluate_layout(batch, fake_layout := self(batch))
        real_pred, real_score, real_raw_scores = self.evaluate_layout(batch, batch.make_struct(self.real_layout_store))

        positive, negative = real_score < fake_score, real_score > fake_score

        if self.hparams.learn_from_critic:
            good_pred = torch.cat([real_pred[positive], fake_pred[negative]])
            bad_pred = torch.cat([fake_pred[positive], real_pred[negative]])
            discriminator_loss = self.adversarial_criterion(encourage=good_pred, discourage=bad_pred)
        else:
            discriminator_loss = self.adversarial_criterion(encourage=real_pred, discourage=fake_pred)
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
        if self.hparams.replace_by_critic:
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

    def on_train_epoch_end(self) -> None:
        self.log_epoch_end(
            total_replacements=sum(self.replacement_counter.values()),
            total_unique_replacements=len(self.replacement_counter)
        )

    def validation_step(self, batch: GraphDrawingData, batch_idx: int):
        # TODO: evaluate scores in validation_epoch_end
        real_layout = batch.make_struct(self.real_layout_store)
        _, real_score, real_raw_scores = self.evaluate_layout(batch, real_layout)
        _, fake_score, fake_raw_scores = self.evaluate_layout(batch, self(batch))
        _, real_human_pref, real_human_raw_scores = self.evaluate_layout(batch, real_layout, self.eval_metric)
        _, fake_human_pref, fake_human_raw_scores = self.evaluate_layout(batch, self(batch), self.eval_metric)
        real_raw_scores.update(real_human_raw_scores)
        fake_raw_scores.update(fake_human_raw_scores)
        score_spc = self.spc(fake_score, real_score)
        human_pref_spc = self.spc(fake_human_pref, real_human_pref)
        raw_scores_spc = {name: self.spc(fake_raw_scores[name], real_raw_scores[name]) for name in fake_raw_scores}
        self.log_evaluation(fake_score.mean().item())
        self.log_val_step(
            score=fake_score.mean().item(),
            score_spc=score_spc.mean().item(),
            human_preference=fake_human_pref.mean().item(),
            human_preference_spc=human_pref_spc.mean().item(),
            **{k: v.mean().item() for k, v in fake_raw_scores.items()},
            **{k + "_spc": v.mean().item() for k, v in raw_scores_spc.items()},
        )

    def test_step(self, batch: GraphDrawingData, batch_idx: int):
        batch.sync_pos(self(batch))
        data_list = batch.to_data_list()
        self.test_step_outputs.append({
            data.name: data.pos.detach().cpu().numpy()
            for data in data_list
        })
