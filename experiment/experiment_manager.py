from smartgd.constants import (
    LIGHTNING_S3_BUCKET, AIMSTACK_UI_URL
)
from .callbacks import SaveMetadata
from .loggers import CustomAimLogger

import os
import textwrap
from typing import Optional, Any, List, Dict

import natsort
from rich.console import Console
from rich.markdown import Markdown
from aim.pytorch_lightning import AimLogger
from lightning_lite.utilities import cloud_io
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, Logger


class ExperimentManager:

    LAST_RUN_HASH_ALIAS = "last"

    def __init__(self, *,
                 experiment_name: Optional[str] = None,
                 experiment_version: Optional[str] = None,
                 experiment_description: Optional[str] = None,
                 run_hash: Optional[str] = None,
                 run_name: Optional[str] = None,
                 run_description: Optional[str] = None):
        # TODO: lookup experiment name and version by `run_hash`
        self._set_up_tensorboard(
            experiment_name=experiment_name,
            experiment_version=experiment_version
        )
        self._set_up_aim(
            experiment_description=experiment_description,
            run_hash=run_hash,
            run_name=run_name,
            run_description=run_description
        )
        self.print_info()
        self._update_metadata()

    def _set_up_tensorboard(self, *, experiment_name, experiment_version):
        self.tensorboard: TensorBoardLogger = TensorBoardLogger(save_dir=f"s3://{LIGHTNING_S3_BUCKET}",
                                                                name=experiment_name,
                                                                version=experiment_version)
        self.fs = cloud_io.get_filesystem(self.log_dir)
        self.experiment_meta_callback = SaveMetadata(dirpath=self.log_dir)

    def _set_up_aim(self, *,
                    experiment_description, run_hash, run_name, run_description):
        self.aim: AimLogger = CustomAimLogger(experiment=f"{self.experiment_name}/{self.experiment_version}",
                                              run_hash=self._resolve_run_hash(run_hash))
        self.experiment_description = experiment_description or (
            self.experiment_metadata["experiment_description"]
            if self.experiment_metadata is not None
            else None
        )
        self.run_name = run_name or self.experiment_version
        self.run_description = run_description or experiment_description
        self.run_meta_callback = SaveMetadata(dirpath=self.checkpoint_dir)

    def _resolve_run_hash(self, run_hash: str):
        if run_hash == self.LAST_RUN_HASH_ALIAS:
            run_hash = self.experiment_metadata["last_run_hash"] if self.experiment_metadata is not None else None
        return run_hash

    def _update_metadata(self):
        self.experiment_meta_callback.update_metadata(dict(
            experiment_name=self.experiment_name,
            experiment_version=self.experiment_version,
            experiment_description=self.experiment_description,
            last_run_hash=self.run_hash
        ))
        self.run_meta_callback.update_metadata(dict(
            run_name=self.run_name,
            run_hash=self.run_hash,
            run_description=self.run_description
        ))

    def print_info(self):
        console = Console()
        console.print(Markdown(textwrap.dedent(f"""
        # SmartGD Experiment
        * **Experiment Name**: {self.experiment_name}
        * **Experiment Version**: {self.experiment_version}
        * **Experiment Description**: {self.experiment_description}
        * **Run Name**: {self.run_name}
        * **Run Hash**: {self.run_hash}
        * **Run Description**: {self.run_description}
        * **Tracking URL**: [{self.tracking_url}]({self.tracking_url})
        """)))

    @property
    def tracking_url(self) -> str:
        return f"{AIMSTACK_UI_URL}/runs/{self.run_hash}"

    @property
    def log_dir(self) -> str:
        return self.tensorboard.log_dir

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.log_dir}/checkpoints/{self.run_hash}"

    @property
    def experiment_name(self) -> str:
        return self.tensorboard.name

    @property
    def experiment_version(self) -> str:
        raw_version = self.tensorboard.version
        return raw_version if isinstance(raw_version, str) else f"version_{raw_version}"

    @property
    def experiment_description(self) -> Optional[str]:
        # return self.aim.experiment.props.experiment_obj.description # this is not supported yet
        return self._experiment_description

    @experiment_description.setter
    def experiment_description(self, value: Optional[str]):
        # if value:
        #     self.aim.experiment.props.experiment_obj.description = value # this is not supported yet
        self._experiment_description = value

    @property
    def experiment_metadata(self) -> Optional[Dict[str, Any]]:
        return self.experiment_meta_callback.load_metadata()

    @property
    def run_hash(self) -> str:
        return self.aim.version

    @property
    def run_name(self) -> str:
        return self.aim.experiment.name

    @run_name.setter
    def run_name(self, value: Optional[str]):
        if value:
            self.aim.experiment.name = value

    @property
    def run_description(self) -> str:
        return self.aim.experiment.description

    @run_description.setter
    def run_description(self, value: Optional[str]):
        if value:
            self.aim.experiment.description = value

    @property
    def run_metadata(self) -> Optional[Dict[str, Any]]:
        return self.run_meta_callback.load_metadata()

    def loggers(self, *additional: Logger) -> List[Logger]:
        managed: List[Logger] = [self.tensorboard, self.aim]
        return managed + list(additional)

    def callbacks(self, *additional: Callback) -> List[Callback]:
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"{{epoch}}-{{step}}",
            # save_top_k=5,
            save_last=True
        )
        # script_callback = ModelScript(
        #     dirpath=f"{self.log_dir}/scripts",
        #     modules=["generator", "discriminator"]
        # )
        managed: List[Callback] = [
            checkpoint_callback,
            self.experiment_meta_callback,
            self.run_meta_callback
        ]
        return managed + list(additional)

    @property
    def last_ckpt_path(self) -> Optional[str]:
        last_ckpts = natsort.natsorted(self.fs.glob(f"{self.checkpoint_dir}/last-v*.ckpt"), reverse=True)
        last_ckpt = f"{self.checkpoint_dir}/last.ckpt"
        if len(last_ckpts) > 0:
            return f"{self.checkpoint_dir}/{os.path.basename(last_ckpts[0])}"
        if self.fs.exists(last_ckpt):
            return f"{self.checkpoint_dir}/{os.path.basename(last_ckpt)}"
        return None

    def trainer_init_args(self, *,
                          loggers: Optional[List[Logger]] = None,
                          callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        return dict(logger=self.loggers(*loggers or []),
                    callbacks=self.callbacks(*callbacks or []))

    def trainer_fit_args(self) -> Dict[str, Any]:
        return dict(ckpt_path=self.last_ckpt_path)
