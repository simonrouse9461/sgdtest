from typing import Mapping

import numpy as np


class RandomLayoutStore(dict):

    def __init__(self, template: Mapping[str, np.ndarray]):
        super().__init__()
        self.template = template

    def __missing__(self, key):
        return np.random.random(self.template[key].shape)
