from smartgd.constants import (
    AIMSTACK_SERVER_URL, TRAIN_PREFIX, VAL_PREFIX, TEST_PREFIX
)

from typing import Optional, Any

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import rank_zero_experiment
from aim.pytorch_lightning import AimLogger
from aim.sdk.run import Run


class CustomAimLogger(AimLogger):

    def __init__(self, *,
                 run_hash: Optional[str] = None,
                 step_metrix_suffix: Optional[str] = "_step",
                 epoch_metrix_suffix: Optional[str] = "_epoch",
                 force_resume: bool = False,
                 **kwargs):
        super().__init__(
            repo=f"aim://{AIMSTACK_SERVER_URL}",
            train_metric_prefix=TRAIN_PREFIX,
            val_metric_prefix=VAL_PREFIX,
            test_metric_prefix=TEST_PREFIX,
            **kwargs
        )

        self._run_hash = run_hash
        self._step_metrix_suffix = step_metrix_suffix
        self._epoch_metrix_suffix = epoch_metrix_suffix
        self._force_resume = force_resume

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
