"""This module provides functionality to run tasks at a fixed frequency."""

from __future__ import annotations

import time
from functools import cache, cached_property

from affctrllib.const import TOL


def timestamp() -> float:
    """Return the time in seconds since the epoch.

    Returns
    -------
    float
        The time in seconds since the epoch.
    """
    # return time.monotonic()
    # return time.perf_counter()
    # return time.process_time()
    # return time.thread_time()
    return time.time()


class Rate(object):
    _frequency: float
    _actual_cycle_time: float
    _start_timer: float
    _start: float

    def __init__(self, frequency: float | int) -> None:
        """Initialize the Rate object.

        Parameters
        ----------
        frequency : float | int
            The frequency to be run periodically.
        """
        self._set_frequency(frequency)
        self._actual_cycle_time = 0.0
        self.start()

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

    @cached_property
    def frequency(self) -> float:
        """Return the frequency.

        Returns
        -------
        float
            The frequency to run a loop in Hertz.
        """
        return self._frequency

    def cycle_time(self) -> float:
        """Return the actual execution time in seconds.

        Return the time in seconds that took for execution. Therefore,
        sleeping time is calculated by substracting this value from
        the expected cycle time, i.e. the inverse of the given
        frequency.

        Returns
        -------
        float
            The actual execution time in seconds.
        """
        return self._actual_cycle_time

    @cache
    def expected_cycle_time(self) -> float:
        """Return the expected cycle time in seconds.

        The value is calculated by inverting the given frequency.

        Returns
        -------
        float
            The expected cycle time for each loop.
        """
        return 1.0 / self._frequency

    def now(self) -> float:
        """Return the time in seconds since the clock started."""
        return timestamp() - self._start_timer

    def start(self) -> float:
        """Restart the clock.

        This is useful when resetting immediately before entering main
        loop after long initialization process.

        Returns
        -------
        float
            The timestamp when the clock started.

        Examples
        --------
        >>> r = Rate(10)
        >>> # Long long initialization process
        >>> r.start()    # Just before entering loop
        >>> while True:
        ...     # Tasks to be executed in each loop
        ...     # do_some_tasks()
        ...     r.sleep()
        """
        self._start_timer = timestamp()
        self._start = self._start_timer
        return self._start_timer

    def reset(self) -> float:
        """Reset the clock for the current loop.

        This function can be used to reset the actual cycle time for
        the current loop due to some reason. It doesn't affect the current
        time elapsed since the clock started.

        Returns
        -------
        float
            The timestamp when the previous loop started.

        Examples
        --------
        >>> r = Rate(10)
        >>> while True:
        ...     # do_some_tasks()  # Unexpected error occurred
        ...     r.reset()
        ...     # do_some_tasks()  # Try again
        ...     r.sleep()
        """
        self._start = timestamp()
        return self._start

    def sleep(self) -> None:
        """Sleep the current process (thread) until the expected cycle time is consumed.

        Examples
        --------
        >>> r = Rate(10)
        >>> while True:
        ...     # Tasks to be executed in each loop
        ...     # do_some_tasks()
        ...     r.sleep()
        """
        expected_end = self._start + self.expected_cycle_time()
        actual_end = timestamp()

        # Detect backward jumps in time.
        if actual_end < self._start:
            expected_end = actual_end + self.expected_cycle_time()

        # Calculate the time we'll sleep for.
        sleep_time = expected_end - actual_end

        # Calculate the actual amount of time the loop took.
        self._actual_cycle_time = actual_end - self._start

        # Reset the start time of the loop.
        self._start = expected_end

        # If we've taken too much time we won't sleep.
        if sleep_time <= 0.0:
            # If we've jumped forward in time, or the loop has taken
            # more than a full extra cycle, reset our cycle.
            if actual_end > expected_end + self.expected_cycle_time():
                self._start = actual_end
            return None

        return time.sleep(sleep_time)
