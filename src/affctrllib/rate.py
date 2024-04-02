"""This module provides functionality to run tasks at a fixed frequency."""

from __future__ import annotations

from affctrllib.const import TOL


class Rate(object):
    _frequency: float

    def __init__(self, frequency: float | int) -> None:
        """Initialize the Rate object.

        Parameters
        ----------
        frequency : float | int
            The frequency to be run periodically.
        """
        self._set_frequency(frequency)

    def _set_frequency(self, frequency: float | int) -> float:
        """Set a frequency.

        Parameters
        ----------
        frequency : float | int
            The frequency to be run periodically.

        Returns
        -------
        float
            The given frequency that is converted into a float.

        Raises
        ------
        ValueError
            If `frequency` is a negative value or very close to zero.
        """
        if frequency <= TOL:
            msg = f"given frequency is unacceptable: negative or very close to zero."
            raise ValueError(msg)
        self._frequency = float(frequency)
        return self._frequency

    @property
    def frequency(self) -> float:
        return self._frequency
