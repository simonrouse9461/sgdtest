import json
from typing import Optional, Any

import pytorch_lightning as L
from lightning_lite.utilities import cloud_io


class SaveMetadata(L.Callback):

    def __init__(self, *, dirpath: str, metadata: Optional[dict[str, Any]] = None):
        super().__init__()
        self.dirpath = dirpath
        self.fs = cloud_io.get_filesystem(dirpath)
        self.metadata = metadata or {}

    @property
    def meta_file(self) -> str:
        return f"{self.dirpath}/metadata.json"

    def update_metadata(self, metadata: dict[str, Any]):
        self.metadata.update(metadata)

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        with self.fs.open(self.meta_file, "w") as fout:
            json.dump(self.metadata, fout)

    def load_metadata(self) -> Optional[dict[str, Any]]:
        if not self.fs.exists(self.meta_file):
            return None
        with self.fs.open(self.meta_file, "r") as fin:
            return json.load(fin)
