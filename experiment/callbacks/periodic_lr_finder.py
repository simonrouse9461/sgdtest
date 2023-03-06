import pytorch_lightning as L
from torch.optim.lr_scheduler import LambdaLR
import aim
from aim.pytorch_lightning import AimLogger


class PeriodicLRFinder(L.callbacks.LearningRateFinder):
    def __init__(self, interval, *args, **kwargs):
        super().__init__(*args,
                         update_attr=True,
                         **kwargs)
        self.interval = interval

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        return

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.current_epoch % self.interval == 0:
            self.lr_find(trainer, pl_module)
            self._log_lr_finder_curve(trainer)
            if isinstance(scheduler := pl_module.lr_schedulers(), LambdaLR):
                scheduler.step()

    def _log_lr_finder_curve(self, trainer: L.Trainer):
        for logger in trainer.loggers:
            if isinstance(logger, AimLogger):
                logger.experiment.track(
                    value=aim.Image(self.optimal_lr.plot(suggest=True)),
                    name="lr_curve",
                    epoch=trainer.current_epoch,
                    context=dict(aggr="epoch")
                )
