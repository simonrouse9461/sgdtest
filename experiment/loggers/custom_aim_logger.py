from smartgd.constants import (
    AIMSTACK_SERVER_URL, TRAIN_PREFIX, VAL_PREFIX, TEST_PREFIX
)
import re
from typing import Optional, Any

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment
from aim.pytorch_lightning import AimLogger
from aim.sdk import Run, Repo


class CustomAimLogger(AimLogger):
    """Custom AimLogger that supports copying from another run.
    example:
    - Create a new run when `run_hash` is None.
    - `run_hash` can be `copy:abc123` to copy from run `abc123`.
    - Use `last` to continue the last run, or `copy:last` to copy from the last run.
    """

    RUN_HASH_ALIAS_LAST = "last"
    RUN_HASH_OP_PREFIX_COPY = "copy"
    RUN_HASH_OP_SEP = ":"

    def __init__(self, *,
                 experiment_name: Optional[str] = None,
                 run_hash: Optional[str] = None,
                 step_metrix_suffix: Optional[str] = "_step",
                 epoch_metrix_suffix: Optional[str] = "_epoch",
                 force_resume: bool = False,
                 **kwargs):
        super().__init__(
            repo=f"aim://{AIMSTACK_SERVER_URL}",
            experiment=experiment_name,
            train_metric_prefix=TRAIN_PREFIX,
            val_metric_prefix=VAL_PREFIX,
            test_metric_prefix=TEST_PREFIX,
            run_hash=run_hash,
            **kwargs
        )
        self._repo = None

        # Lookup experiment group and version by `run_hash` if `run_hash` is not None.
        self._run_hash, self._copy_from = self._parse_run_hash(run_hash).values()
        self._run_hash = self._resolve_run_hash(self._run_hash)
        self._copy_from = self._resolve_run_hash(self._copy_from)
        self._experiment_name = self._resolve_experiment_name()

        self._step_metrix_suffix = step_metrix_suffix
        self._epoch_metrix_suffix = epoch_metrix_suffix
        self._force_resume = force_resume

        if self._copy_from and (hparams := self.repo.get_run(self._copy_from).get('hparams')):
            self.experiment['hparams'] = hparams

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            self._repo = Repo(path=self._repo_path)
        return self._repo

    def _parse_run_hash(self, run_hash: Optional[str]) -> dict[str, Any]:
        copy_prefix = self.RUN_HASH_OP_PREFIX_COPY + self.RUN_HASH_OP_SEP
        copy_from = None
        if run_hash and run_hash.startswith(copy_prefix):
            copy_from = run_hash[len(copy_prefix):]
            run_hash = None
        return dict(run_hash=run_hash, copy_from=copy_from)

    def _resolve_run_hash(self, run_hash: Optional[str]) -> str:
        if run_hash == self.RUN_HASH_ALIAS_LAST:
            assert self._experiment_name is not None, (
                f"Experiment group and name must be specified when using alias '{self.RUN_HASH_ALIAS_LAST}'."
            )
            runs = filter(lambda run: run.experiment == self._experiment_name, self.repo.iter_runs())
            runs = sorted(runs, key=lambda run: run.creation_time)
            assert runs, f"No runs found for experiment '{self._experiment_name}'."
            return runs[-1].hash
        return run_hash

    def _resolve_experiment_name(self):
        assert not (self._copy_from and self._run_hash)
        if src_hash := self._copy_from or self._run_hash:
            src_run = self.repo.get_run(src_hash)
            experiment_name = src_run.experiment
        else:
            assert self._experiment_name is not None, (
                "Experiment group and name must be specified when `run_hash` is None."
            )
            experiment_name = self._experiment_name
        if self._experiment_name is not None:
            assert self._experiment_name == experiment_name, (
                f"Experiment name='{self._experiment_name}' does not match "
                f"the original experiment name '{experiment_name}'."
            )
        return experiment_name

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        if self._run is None:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                    capture_terminal_logs=self._capture_terminal_logs,
                    force_resume=self._force_resume
                )
            else:
                self._run = Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                    force_resume=self._force_resume
                )
                self._run_hash = self._run.hash
        return self._run

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metric_items: dict[str: Any] = {k: v for k, v in metrics.items()}

        epoch: Optional[int] = None
        if 'epoch' in metric_items:
            epoch: int = metric_items.pop('epoch')

        for k, v in metric_items.items():
            name = k
            context = {}
            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = name.removeprefix(self._train_metric_prefix)
                context['subset'] = 'train'
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = name.removeprefix(self._test_metric_prefix)
                context['subset'] = 'test'
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = name.removeprefix(self._val_metric_prefix)
                context['subset'] = 'val'

            if self._step_metrix_suffix and name.endswith(self._step_metrix_suffix):
                name = name.removesuffix(self._step_metrix_suffix)
                context['aggr'] = 'step'
            elif self._epoch_metrix_suffix and name.endswith(self._epoch_metrix_suffix):
                name = name.removesuffix(self._epoch_metrix_suffix)
                context['aggr'] = 'epoch'

            self.experiment.track(v, name=name, step=step, epoch=epoch, context=context)

    @property
    def name(self) -> str:
        return self.experiment.experiment
