from smartgd.experiment.utils import TorchScriptUtils

from torch import jit
import lightning as L
from lightning_lite.utilities import cloud_io


class ModelScript(L.Callback):

    def __init__(self, *, dirpath: str, modules: list[str]):
        super().__init__()
        self.dirpath = dirpath
        self.fs = cloud_io.get_filesystem(dirpath)
        self.modules = modules

    @property
    def script_globs(self):
        return {module: f"{self.dirpath}/{module}.*.pt" for module in self.modules}

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        raise Exception
        for module, script_glob in self.script_globs.items():
            script_paths = self.fs.glob(script_glob)
            match len(script_paths):
                case 0:
                    model = getattr(pl_module, module)
                    script = model if isinstance(model, jit.ScriptModule) else jit.script(model)
                    digest = TorchScriptUtils.digest_script(script)
                    script_path = f"{self.dirpath}/{module}.{digest}.pt"
                    print(f"Saving model definition for '{module}' to '{script_path}'...")
                    with self.fs.open(script_path, "wb") as fout:
                        jit.save(script, fout)
                case 1:
                    with self.fs.open(script_paths[0], "rb") as fin:
                        print(f"Found existing model definition at '{script_paths[0]}' for '{module}'. Overwriting...")
                        setattr(pl_module, module, jit.load(fin))
                case _:
                    raise Exception("More than one model version found!")
