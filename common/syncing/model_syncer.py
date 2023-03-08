from smartgd.constants import MODEL_S3_BUCKET
from ..jittools import TorchScriptUtils

from typing import Optional, Callable, Any
import json
import datetime
import tempfile
import pickle
import dataclasses

import torch
from torch import nn
from torch import jit
from lightning_lite.utilities import cloud_io


class ModelSyncer:

    DEFAULT_MODEL_NAME = "__model__"
    LATEST_VERSION_NAME = "__latest__"
    DEFAULT_CONFIG_NAME = "__default__"

    def __init__(self, uri: str = f"s3://{MODEL_S3_BUCKET}"):
        self.uri = uri
        self.fs = cloud_io.get_filesystem(uri)

    def model_base_dir(self, name: str) -> str:
        return f"{self.uri}/{name}"

    def model_version_dir(self, name: str, version: str) -> str:
        # str_repr = json.dumps(self._preprocess_args(arg), default=str)
        # hash = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
        return f"{self.model_base_dir(name)}/{version}"

    def model_file_path(self, name: str, version: str, file: str) -> str:
        return f"{self.model_version_dir(name, version)}/{file}"

    def model_script_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "script.pt")

    def model_tree_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "tree.txt")

    def model_graph_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "graph.txt")

    def model_code_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "code.py")

    def model_args_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "args.pkl")

    def model_meta_path(self, name: str, version: str) -> str:
        return self.model_file_path(name, version, "meta.json")

    def exists(self, name: str, version: Optional[str] = None) -> bool:
        if version is None:
            version = self.LATEST_VERSION_NAME
        return self.fs.exists(self.model_script_path(name, version))

    @staticmethod
    def _serialize_model(model: nn.Module) -> jit.ScriptModule:
        model_script = model if isinstance(model, jit.ScriptModule) else jit.script(model)
        # Force serialize and deserialize to yield consistent digest value
        tmp = tempfile.mktemp()
        jit.save(model_script, tmp)
        return jit.load(tmp)

    def _save_version(self, script: torch.ScriptModule, name: str, version: str, args: dict, metadata: dict):
        with self.fs.open(self.model_script_path(name, version), "wb") as fout:
            jit.save(script, fout)
        with self.fs.open(self.model_tree_path(name, version), "w") as fout:
            fout.write(TorchScriptUtils.format_tree(script))
        with self.fs.open(self.model_graph_path(name, version), "w") as fout:
            fout.write(TorchScriptUtils.format_graph(script))
        with self.fs.open(self.model_code_path(name, version), "w") as fout:
            fout.write(TorchScriptUtils.format_code(script))
        with self.fs.open(self.model_args_path(name, version), "wb") as fout:
            pickle.dump(args, fout)
        with self.fs.open(self.model_meta_path(name, version), "w") as fout:
            json.dump(metadata, fout)

    def save(self,
             model: nn.Module,
             *,
             name: Optional[str] = None,
             version: Optional[str | list[str]] = None,
             configuration: Optional[str] = None,  # TODO
             latest: bool = True,
             metadata: Optional[dict] = None) -> dict:
        if name is None:
            name = self.DEFAULT_MODEL_NAME
        if version is None:
            version = []
        if isinstance(version, str):
            version = [version]
        model_script = self._serialize_model(model)
        digest = TorchScriptUtils.digest_script(model_script)
        metadata = {
            "model_name": name,
            "timestamp": str(datetime.datetime.utcnow()),
            "class_name": model_script.original_name,
            "md5_digest": digest
        } | (metadata or {})
        version.append(digest)
        if latest:
            version.append(self.LATEST_VERSION_NAME)
        for version_name in version:
            self._save_version(model_script, name, version_name, dataclasses.asdict(model), metadata)
        return metadata

    def _check_before_load(self,
                           name: Optional[str] = None,
                           version: Optional[str] = None) -> tuple[str, str]:
        if name is None:
            name = self.DEFAULT_MODEL_NAME
        if version is None:
            version = self.LATEST_VERSION_NAME
        assert self.exists(name=name, version=version), f"Model '{name}:{version}' does not exist!"
        return name, version

    def load(self, *,
             name: Optional[str] = None,
             version: Optional[str] = None) -> jit.ScriptModule:
        name, version = self._check_before_load(name, version)
        print(f"Loading model definition '{name}/{version}'...")
        with self.fs.open(self.model_script_path(name, version), "rb") as fin:
            return jit.load(fin)

    def load_metadata(self, *,
                      name: Optional[str] = None,
                      version: Optional[str] = None) -> dict:
        name, version = self._check_before_load(name, version)
        with self.fs.open(self.model_meta_path(name, version), "r") as fin:
            return json.load(fin)

    def load_arguments(self, *,
                       name: Optional[str] = None,
                       version: Optional[str] = None,
                       serialization: Callable[[Any], Any] = None) -> dict:
        name, version = self._check_before_load(name, version)
        with self.fs.open(self.model_args_path(name, version), "rb") as fin:
            args = pickle.load(fin)
        if serialization is not None:
            args = json.loads(json.JSONEncoder(default=serialization).encode(args))
        return args

    def create_alias(self, *,
                     name: Optional[str] = None,
                     version: Optional[str] = None,
                     new_name: Optional[str] = None,
                     new_version: Optional[str | list[str]] = None,
                     latest: bool = False) -> dict:
        if new_name is None:
            new_name = name
        model = self.load(name=name, version=version)
        args = self.load_arguments(name=name, version=version)
        meta = self.load_metadata(name=name, version=version)
        return self.save(model=model,
                         name=new_name,
                         version=new_version,
                         latest=latest,
                         args=args,
                         metadata=meta)
