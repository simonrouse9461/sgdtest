from smartgd.common.data import (
    GraphStruct,
    GraphDrawingData,
)
from smartgd.common.nn import (
    BaseTransformation,
    RescaleByStress,
    Compose,
    Center,
    NormalizeRotation,
    CompositeCritic,
    RGANCriterion,
    BaseAdverserialCriterion,
    SPC
)
from ..data_adaptors import DiscriminatorDataAdaptor, GeneratorDataAdaptor
from ..utils import RandomLayoutStore
from .base_lightning_module import BaseLightningModule

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any, Union, TypeVar

import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
import torch_geometric as pyg
from tqdm.auto import tqdm


LayoutDict = dict[str, np.ndarray]


class SmartGDLightningModule(BaseLightningModule):

    @dataclass
    class Config:
        dataset_name:               str = "Rome"
        generator_spec:             Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        discriminator_spec:         Union[Optional[str], tuple[Optional[str], Optional[str]]] = None
        criteria:                   Union[str, dict[str, float]] = "stress_only"
        train_optional_data_fields: list[str] = field(default_factory=list)
        eval_optional_data_fields:  list[str] = field(default_factory=list)
        init_layout_method:         Optional[str] = "pmds"
        real_layout_candidates:     Union[str, list[str]] = field(default_factory=lambda: [
            "neato", "sfdp", "spring", "spectral", "kamada_kawai", "fa2", "pmds"
        ])
        benchmark_layout_methods:   list[str] = field(default_factory=lambda: [
            "neato", "sfdp", "spring", "spectral", "kamada_kawai", "fa2", "pmds", "sgd2"
        ])
        self_challenging:           bool = True
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
        self.init_layout_store: Optional[LayoutDict] = None
        self.real_layout_store: Optional[LayoutDict] = None
        self.fake_layout_store: LayoutDict = {}
        self.benchmark_layout_stores: Optional[dict[str, LayoutDict]] = None
        self.replacement_counter: Optional[dict[str, int]] = None

        # Functions
        self.critic: Optional[CompositeCritic] = None
        self.val_metric: CompositeCritic = CompositeCritic.from_preset("human_preference")
        self.adversarial_criterion: BaseAdverserialCriterion = RGANCriterion()
        self.canonicalize: BaseTransformation = Compose(
            Center(),
            NormalizeRotation(),
            RescaleByStress()
        )
        self.spc: SPC = SPC(batch_reduce=None)

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
            train_optional_data_fields=config.train_optional_data_fields,
            eval_optional_data_fields=config.eval_optional_data_fields,
            init_layout_method=config.init_layout_method,
            real_layout_candidates=config.real_layout_candidates,
            benchmark_layout_methods=config.benchmark_layout_methods,
            self_challenging=config.self_challenging,
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
        layout_params = self.generate_real_layout_params()
        if self.layout_syncer.exists(**layout_params):
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
        real_layout_metadata = {}
        print("Generating real layouts...")
        for data in tqdm(dataset, desc="Generate Real"):
            data = data.post_transform(self.hparams.static_transform)
            scores = torch.cat([
                # TODO: merge data into batch first
                critic(self.canonicalize(data.make_struct(store, post_transform=self.hparams.dynamic_transform)))
                for layout, store in layout_stores.items()
            ])
            best_layout = list(layout_stores)[scores.argmin()]
            real_layout_store[data.name] = layout_stores[best_layout][data.name]
            real_layout_metadata[data.name] = dict(method=best_layout)
        if not self.layout_syncer.exists(**layout_params):
            print("Uploading real layouts...")
            self.layout_syncer.save(
                real_layout_store,
                metadata=real_layout_metadata,
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
        self.critic = CompositeCritic(
            criteria_weights=self.hparams.criteria_weights,
            batch_reduce=None
        )
        real_layout_params = self.generate_real_layout_params()
        assert self.layout_syncer.exists(**real_layout_params), "Layout not found!"
        self.real_layout_store = self.layout_syncer.load(**real_layout_params)
        if self.hparams.init_layout_method:
            self.init_layout_store = self.layout_syncer.load(name=self.hparams.init_layout_method)
        else:
            self.init_layout_store = RandomLayoutStore(template=self.layout_syncer.load(name="random"))
        self.replacement_counter = defaultdict(int)
        if stage == "test":
            self.setup_test()

    def setup_test(self) -> None:
        self.benchmark_layout_stores = {
            method: self.layout_syncer.load(name=method)
            for method in self.hparams.benchmark_layout_methods
        }
        self.benchmark_layout_stores["real"] = self.layout_syncer.load(
            name="ranked", params=dict(
                candidates=self.hparams.real_layout_candidates,
                criteria=self.hparams.criteria_weights
            )
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.load_generator(checkpoint["hyper_parameters"]["generator"])
        self.load_discriminator(checkpoint["hyper_parameters"]["discriminator"])
        self.real_layout_store = checkpoint["real_layout_store"]
        self.replacement_counter = checkpoint["replacement_counter"]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint["real_layout_store"] = self.real_layout_store
        checkpoint["replacement_counter"] = self.replacement_counter

    def on_train_start(self) -> None:
        GraphDrawingData.set_optional_fields(self.hparams.train_optional_data_fields)

    def on_validation_start(self) -> None:
        GraphDrawingData.set_optional_fields(self.hparams.eval_optional_data_fields)

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
        fake_layout = batch.transform_struct(self.canonicalize, self(batch))
        fake_pred = self.discriminator(fake_layout)
        fake_score, fake_raw_scores = self.critic(fake_layout), self.critic.get_raw_scores()

        real_layout = batch.make_struct(self.real_layout_store)
        real_layout = batch.transform_struct(self.canonicalize, real_layout)
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
        if self.hparams.self_challenging:
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

    def training_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        self.log_epoch_end(
            total_replacements=sum(self.replacement_counter.values()),
            total_unique_replacements=len(self.replacement_counter)
        )

    def validation_step(self, batch: GraphDrawingData, batch_idx: int):
        real_layout = batch.make_struct(self.layout_syncer.load(**self.generate_real_layout_params()))
        real_layout = batch.transform_struct(self.canonicalize, real_layout)
        fake_layout = batch.transform_struct(self.canonicalize, self(batch))
        fake_score = self.critic(fake_layout)
        fake_human_score, fake_raw_scores = self.val_metric(fake_layout), self.val_metric.get_raw_scores()
        real_score = self.critic(real_layout)
        real_human_score, real_raw_scores = self.val_metric(real_layout), self.val_metric.get_raw_scores()
        score_spc = self.spc(fake_score, real_score)
        human_score_spc = self.spc(fake_human_score, real_human_score)
        raw_scores_spc = {name: self.spc(fake_raw_scores[name], real_raw_scores[name]) for name in fake_raw_scores}
        self.log_val_step(
            score=fake_score.mean().item(),
            score_spc=score_spc.mean().item(),
            human_preference_score=fake_human_score.mean().item(),
            human_preference_score_spc=human_score_spc.mean().item(),
            **{k: v.mean().item() for k, v in fake_raw_scores.items()},
            **{k + "_spc": v.mean().item() for k, v in raw_scores_spc.items()},
        )

    def test_step(self, batch: GraphDrawingData, batch_idx: int) -> dict:
        # TODO: use mean metric
        scores = {}
        fake_layout = batch.transform_struct(self.canonicalize, self(batch))
        scores["fake"] = self.critic(fake_layout)
        for method, layout_store in self.benchmark_layout_stores.items():
            layout = batch.make_struct(layout_store)
            layout = batch.transform_struct(self.canonicalize, layout)
            scores[method] = self.critic(layout)
        return scores

    def test_epoch_end(self, outputs: list[dict]):
        scores_list = defaultdict(list)
        for scores in outputs:
            for method, score in scores.items():
                scores_list[method].append(score)
        scores = {
            method: torch.cat(score_list, dim=0)
            for method, score_list in scores_list.items()
        }
        self.evaluations = {
            method: (
                score.mean().item(), score.std().item(),
                (spc := self.spc(scores["fake"], score)).mean().item(), spc.std().item()
            )
            for method, score in scores.items()
        }

