from smartgd.constants import (
    CKPT_S3_BUCKET, AIMSTACK_UI_URL
)
from . import BaseLightningModule, SaveMetadata, UploadPredictions, CustomAimLogger

import os
import textwrap
from typing import Optional, Any, Iterable
from pprint import pformat

import natsort
from deepdiff import DeepDiff
from deepdiff.operator import BaseOperator
from rich.console import Console
from rich.markdown import Markdown
from aim.pytorch_lightning import AimLogger
from lightning_lite.utilities import cloud_io
# TODO: import lightning as L
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger


# TODO implement context manager
class ExperimentManager:

    aim: AimLogger

    def __init__(self, *,
                 experiment_group: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 experiment_description: Optional[str] = None,
                 run_hash: Optional[str] = None,
                 run_name: Optional[str] = None,
                 run_description: Optional[str] = None,
                 pretrain_run_hash: Optional[str] = None,
                 force_resume: bool = False):
        experiment_group = experiment_group or "default_experiment"
        self.aim = CustomAimLogger(experiment_name=(f"{experiment_group}.{experiment_name}"
                                                    if experiment_group and experiment_name else None),
                                   run_hash=run_hash,
                                   force_resume=force_resume)
        self.fs = cloud_io.get_filesystem(self.checkpoint_dir)
        self.pretrain_run_hash = pretrain_run_hash

        if experiment_description:
            self.experiment_description = experiment_description
        if run_name:
            self.run_name = run_name
        if run_description:
            self.run_description = run_description

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="{epoch}-{step}-{evaluation}",
            monitor="evaluation",
            mode="min",
            save_top_k=5,
            save_last=True
        )

        self.prediction_callback = UploadPredictions(
            run_hash=self.run_hash
        )

        # TODO: save metadata

        self.print_info()

    def print_info(self):
        console = Console()
        console.print(Markdown(textwrap.dedent(f"""
        # SmartGD Experiment
        * **Experiment Group**: `{self.experiment_group}`
        * **Experiment Name**: `{self.experiment_name}`
        * **Experiment Description**: {self.experiment_description}
        * **Run Hash**: `{self.run_hash}`
        * **Run Name**: `{self.run_name}`
        * **Run Description**: {self.run_description}
        * **Tracking URL**: [{self.tracking_url}]({self.tracking_url})
        """)))

    @property
    def tracking_url(self) -> str:
        return f"http://{AIMSTACK_UI_URL}/runs/{self.run_hash}"

    @property
    def checkpoint_dir(self) -> str:
        return f"s3://{CKPT_S3_BUCKET}/{self.run_hash}"

    @property
    def pretrain_checkpoint_dir(self) -> Optional[str]:
        if self.pretrain_run_hash:
            return f"s3://{CKPT_S3_BUCKET}/{self.pretrain_run_hash}"
        return None

    @property
    def experiment_group(self) -> str:
        return self.aim.name.split(".")[0]

    @property
    def experiment_name(self) -> str:
        return self.aim.name.split(".")[1]

    @property
    def experiment_description(self) -> Optional[str]:
        # TODO: this is not supported for remote repo yet
        try:
            return self.aim.experiment.props.experiment_obj.description
        except Exception as e:
            return f'{type(e).__name__}: {e}'

    @experiment_description.setter
    def experiment_description(self, value: str):
        # TODO: this is not supported for remote repo yet
        self.aim.experiment.props.experiment_obj.description = value

    @property
    def run_hash(self) -> str:
        return self.aim.version

    @property
    def run_name(self) -> str:
        return self.aim.experiment.name

    @run_name.setter
    def run_name(self, value: Optional[str]):
        self.aim.experiment.name = value

    @property
    def run_description(self) -> str:
        return self.aim.experiment.description

    @run_description.setter
    def run_description(self, value: str):
        self.aim.experiment.description = value

    @property
    def hyperparameters(self) -> Optional[dict[str, Any]]:
        return self.aim.experiment.get('hparams')

    @property
    def loggers(self) -> list[Logger]:
        return [self.aim]

    @property
    def callbacks(self) -> list[Callback]:
        return [
            self.checkpoint_callback,
            self.prediction_callback,
        ]

    def _get_last_ckpt(self, ckpt_root: str) -> Optional[str]:
        last_ckpts = natsort.natsorted(self.fs.glob(f"{ckpt_root}/last-v*.ckpt"), reverse=True)
        last_ckpt = f"{ckpt_root}/last.ckpt"
        if len(last_ckpts) > 0:
            return f"{ckpt_root}/{os.path.basename(last_ckpts[0])}"
        if self.fs.exists(last_ckpt):
            return f"{ckpt_root}/{os.path.basename(last_ckpt)}"
        return None

    @property
    def last_ckpt_path(self) -> Optional[str]:
        return self._get_last_ckpt(self.checkpoint_dir)

    def create_module(self, module_cls: type[BaseLightningModule], **kwargs) -> BaseLightningModule:
        class ListTupleMatchOperator(BaseOperator):
            def give_up_diffing(self, level, diff_instance):
                if isinstance(level.t1, tuple):
                    level.t1 = list(level.t1)
                if isinstance(level.t2, tuple):
                    level.t2 = list(level.t2)
        if self.pretrain_checkpoint_dir:
            module = module_cls.load_from_checkpoint(
                checkpoint_path=self._get_last_ckpt(self.pretrain_checkpoint_dir),
                **kwargs
            )
        else:
            module = module_cls(**kwargs)
        if self.hyperparameters:
            diff = DeepDiff(
                t1=self.hyperparameters, t2=dict(module.hparams),
                custom_operators=[ListTupleMatchOperator(types=[Iterable])]
            )
            assert not diff, f"Hyperparameters mismatch:\n{pformat(diff.to_dict())}"
            module.load_hyperparameters(self.hyperparameters)
        return module

    def trainer_init_args(self, *,
                          loggers: Optional[list[Logger]] = None,
                          callbacks: Optional[list[Callback]] = None) -> dict[str, Any]:
        return dict(logger=self.loggers + (loggers or []),
                    callbacks=self.callbacks + (callbacks or []))

    def trainer_run_args(self) -> dict[str, Any]:
        return dict(ckpt_path=self.last_ckpt_path)

    def close(self) -> None:
        self.aim.experiment.close()
