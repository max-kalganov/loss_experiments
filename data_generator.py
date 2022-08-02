""" Generates simple dataset with two hidden parameters"""
from typing import Optional, Callable, Tuple
import numpy as np
import gin


def lin_key_function(x1, x2, w1, w2):
    return 1 / (1 + np.exp(- (w1 * x1 + w2 * x2)))


@gin.configurable()
def get_dataset(w1: float, w2: float, samples_num: int,
                key: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    key = key if key is not None else lin_key_function

    x = np.random.random(size=(samples_num, 2))
    y = key(x[:, 0], x[:, 1], w1, w2)
    return x, y
