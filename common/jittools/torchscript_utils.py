import re
import hashlib

from torch import jit


class TorchScriptUtils:

    @classmethod
    def format_code(cls, script: jit.ScriptModule, verbose: bool = False):
        def get_all_codes(model):
            if hasattr(model, "code"):
                yield model.original_name, model.code
            elif verbose:
                print(f"No code available for {model.original_name}.")
            for child in model.children():
                yield from get_all_codes(child)
        code_list = [re.sub(r"(?<=def )(?=forward)", f"_{name}__", code)
                     for name, code in dict(get_all_codes(script)).items()]
        return "\n".join(code_list)

    @classmethod
    def format_tree(cls, script: jit.ScriptModule):
        tree_str = str(script)
        tree_str = re.sub(r"Jittable_\w+", "", tree_str)
        return tree_str

    @classmethod
    def format_graph(cls, script: jit.ScriptModule):
        script_str = str(script.inlined_graph)
        script_str = re.sub(r"#.*(?=\n|$)", "", script_str)
        script_str = re.sub(r"\.___torch_mangle_\d+", "", script_str)
        script_str = re.sub(r"Jittable_\w+", "", script_str)
        return script_str

    @classmethod
    def digest_script(cls, script: jit.ScriptModule):
        str_repr = cls.format_graph(script)
        return hashlib.md5(str_repr.encode("utf-8")).hexdigest()
