from collections import deque
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, np.ndarray)


class Filter(Generic[T]):
    _n_points: int
    _x_buffer: deque[T | float]
    _y_prev: T

    def __init__(self, n_points: int | None = None) -> None:
        self._n_points = 5
        if n_points is not None:
            self.n_points = n_points

        self._x_buffer = deque()
        for _ in range(self.n_points):
            self._x_buffer.append(0.0)
        self._y_prev = 0.0

    @property
    def n_points(self) -> int:
        return self._n_points

    def set_n_points(self, n_points: int) -> None:
        self._n_points = n_points

    @n_points.setter
    def n_points(self, n_points) -> None:
        self.set_n_points(n_points)

    def update(self, x: T) -> T:
        self._x_buffer.append(x)
        self._y_prev = self._y_prev + x - self._x_buffer.popleft()
        return self._y_prev / self.n_points
