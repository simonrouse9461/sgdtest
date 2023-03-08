from smartgd.constants import (
    AIMSTACK_SERVER_URL, TRAIN_PREFIX, VAL_PREFIX, TEST_PREFIX
)

from typing import Optional, Any, Dict

from pytorch_lightning.utilities import rank_zero_only
from aim.pytorch_lightning import AimLogger


class CustomAimLogger(AimLogger):

    def __init__(self, *,
                 run_hash: Optional[str] = None,
                 step_metrix_suffix: Optional[str] = "_step",
                 epoch_metrix_suffix: Optional[str] = "_epoch",
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

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metric_items: Dict[str: Any] = {k: v for k, v in metrics.items()}

        epoch: Optional[int] = None
        if 'epoch' in metric_items:
            epoch: int = metric_items.pop('epoch')

        for k, v in metric_items.items():
            name = k
            context = {}
            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = self.removeprefix(name, self._train_metric_prefix)
                context['subset'] = 'train'
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = self.removeprefix(name, self._test_metric_prefix)
                context['subset'] = 'test'
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = self.removeprefix(name, self._val_metric_prefix)
                context['subset'] = 'val'

            if self._step_metrix_suffix and name.endswith(self._step_metrix_suffix):
                name = self.removesuffix(name, self._step_metrix_suffix)
                context['aggr'] = 'step'
            elif self._epoch_metrix_suffix and name.endswith(self._epoch_metrix_suffix):
                name = self.removesuffix(name, self._epoch_metrix_suffix)
                context['aggr'] = 'epoch'

            self.experiment.track(v, name=name, step=step, epoch=epoch, context=context)

    def removeprefix(self, text: str, prefix: str):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    def removesuffix(self, text: str, suffix: str):
        if text.endswith(suffix):
            return text[:-len(suffix)]
        return text
