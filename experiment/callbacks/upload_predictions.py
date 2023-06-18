from smartgd.common.jittools import TorchScriptUtils

from functools import reduce
from typing import List

from torch import jit
import pytorch_lightning as L


class UploadPredictions(L.Callback):

    def __init__(self, *, run_hash: str):
        super().__init__()
        self.run_hash = run_hash
        self.predictions = None

    def on_test_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        self.predictions = reduce(lambda a, b: a | b, module.test_step_outputs)

    def on_test_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        # TODO: should not reuse layout_syncer from module
        module.layout_syncer.batch_update(name=self.run_hash, layout_dict=self.predictions)

