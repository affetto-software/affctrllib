import numpy as np
import pytest
from affctrllib.filter import Filter
from numpy.testing import assert_array_equal


class TestFilter:
    def test_init(self) -> None:
        filt = Filter()
        assert filt.n_points == 5

    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_set_n_points(self, n) -> None:
        filt = Filter()
        filt.set_n_points(n)
        assert filt.n_points == n

    @pytest.mark.parametrize("n", [5, 7, 9, 11])
    def test_n_points_setter(self, n) -> None:
        filt = Filter()
        filt.n_points = n
        assert filt.n_points == n

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_set_n_points_in_init(self, n) -> None:
        filt = Filter(n)
        assert filt.n_points == n

    def test_update_int(self) -> None:
        input_signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        n_points = 3
        expected = [1 / 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        filt = Filter(n_points)
        output_signal = []
        for x in input_signal:
            y = filt.update(x)
            output_signal.append(y)
        assert output_signal == expected

    def test_update_array(self) -> None:
        input_signal = [
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([3, 3, 3]),
            np.array([4, 4, 4]),
            np.array([5, 5, 5]),
            np.array([6, 6, 6]),
            np.array([7, 7, 7]),
        ]
        n_points = 3
        expected = [
            np.array([1 / 3, 1 / 3, 1 / 3]),
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([3, 3, 3]),
            np.array([4, 4, 4]),
            np.array([5, 5, 5]),
            np.array([6, 6, 6]),
        ]
        filt = Filter(n_points)
        output_signal = []
        for x in input_signal:
            y = filt.update(x)
            output_signal.append(y)
        for i in range(7):
            assert_array_equal(output_signal[i], expected[i])
