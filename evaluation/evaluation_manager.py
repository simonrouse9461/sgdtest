

class EvaluationManager:


    def __init__(self):
        ...

    # def

    def setup_test(self) -> None:
        self.benchmark_layout_stores = {
            method[0] if method[1] is None else f"{method[0]}:{method[1]['metric']}":
                self.layout_syncer.load(name=method[0], params=method[1])
            for method in self.hparams.benchmark_layout_methods
        }
        self.benchmark_layout_stores["real"] = self.layout_syncer.load(**self.generate_real_layout_params())
