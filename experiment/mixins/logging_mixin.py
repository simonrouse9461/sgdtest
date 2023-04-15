from smartgd.constants import TRAIN_PREFIX, VAL_PREFIX, TEST_PREFIX

from abc import ABC, abstractmethod


class LoggingMixin(ABC):

    @abstractmethod
    def log_dict(self, *args, **kwargs):
        return NotImplemented

    def log_evaluation(self, value):
        self.log_dict(dictionary=dict(evaluation=value),
                      on_step=False,
                      on_epoch=True,
                      logger=True)

    def log_with_prefix(self, *,
                        prefix: str,
                        dictionary: dict,
                        on_step: bool,
                        on_epoch: bool,
                        **kwargs):
        suffix = ""  # TODO: subclass LightningModule to implement this logic
        if on_step and not on_epoch:
            suffix = "_step"
        if on_epoch and not on_step:
            suffix = "_epoch"
        prefixed_dict = {prefix + k + suffix: v for k, v in dictionary.items()}
        self.log_dict(dictionary=prefixed_dict,
                      on_step=on_step,
                      on_epoch=on_epoch,
                      logger=True,
                      **kwargs)

    def log_epoch_end(self, **dictionary):
        self.log_with_prefix(prefix="",
                             dictionary=dictionary,
                             on_step=False,
                             on_epoch=True)

    def log_train_step(self, **dictionary):
        self.log_with_prefix(prefix=TRAIN_PREFIX,
                             dictionary=dictionary,
                             on_step=True,
                             on_epoch=True)

    def log_train_step_sum_on_epoch_end(self, **dictionary):
        self.log_with_prefix(prefix=TRAIN_PREFIX,
                             dictionary=dictionary,
                             on_step=True,
                             on_epoch=True,
                             reduce_fx="sum")

    def log_val_step(self, **dictionary):
        self.log_with_prefix(prefix=VAL_PREFIX,
                             dictionary=dictionary,
                             on_step=False,
                             on_epoch=True)

    def log_test_step(self, **dictionary):
        self.log_with_prefix(prefix=TEST_PREFIX,
                             dictionary=dictionary,
                             on_step=False,
                             on_epoch=True)
