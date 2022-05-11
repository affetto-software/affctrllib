import numpy as np
import pytest
from affctrllib.ptp import (
    PTP,
    FifthDegreePolynomialProfile,
    SinusoidalVelocityProfile,
    TrapezoidalVelocityProfile,
    TriangularVelocityProfile,
)
from numpy.testing import assert_array_almost_equal, assert_array_equal

PIx2 = 2.0 * np.pi


class TestPTP:
    def test_init(self) -> None:
        ptp = PTP(0, 1, 5)
        assert ptp.q0 == 0
        assert ptp.qF == 1
        assert ptp.T == 5
        assert ptp.t0 == 0
        assert isinstance(ptp.profile, TriangularVelocityProfile)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("triangular velocity", TriangularVelocityProfile),
            ("triangular", TriangularVelocityProfile),
            ("tri", TriangularVelocityProfile),
            ("sinusoidal velocity", SinusoidalVelocityProfile),
            ("sinusoidal", SinusoidalVelocityProfile),
            ("sin", SinusoidalVelocityProfile),
            ("5th-degree polynomial", FifthDegreePolynomialProfile),
            ("5th degree polynomial", FifthDegreePolynomialProfile),
            ("5th-degree", FifthDegreePolynomialProfile),
            ("5th degree", FifthDegreePolynomialProfile),
            ("5th", FifthDegreePolynomialProfile),
        ],
    )
    def test_select_profile(self, name, expected) -> None:
        ptp = PTP(0, 1, 5, profile_name=name)
        assert isinstance(ptp.profile, expected)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("tra", TrapezoidalVelocityProfile),
            ("trapez", TrapezoidalVelocityProfile),
            ("trapezoidal", TrapezoidalVelocityProfile),
            ("trapezoidal velocity", TrapezoidalVelocityProfile),
        ],
    )
    def test_select_profile_trapezoidal(self, name, expected) -> None:
        ptp = PTP(0, 1, 5, profile_name=name, vmax=0.25)
        assert isinstance(ptp.profile, expected)

    @pytest.mark.parametrize("name", ["hoge", "poly", "tria", "5th ordre"])
    def test_error_invalid_profile_name(self, name) -> None:
        with pytest.raises(ValueError) as excinfo:
            _ = PTP(0, 1, 5, profile_name=name)
        assert f"Invalid profile name: {name}" in str(excinfo.value)

    def test_update_triangular_velocity(self) -> None:
        ptp = PTP(0, 1, 5, profile_name="tri")
        expectation_table = [
            # t, q, dq
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0.08, 0.16),
            (2.5, 0.5, 0.4),
            (4, 0.92, 0.16),
            (5, 1, 0),
            (6, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert ptp.q(t) == q
            assert ptp.dq(t) == dq

    def test_update_triangular_velocity_t0(self) -> None:
        ptp = PTP(0, 1, 5, 1, profile_name="tri")
        expectation_table = [
            # t, q, dq
            (0, 0, 0),
            (1, 0, 0),
            (2, 0.08, 0.16),
            (3.5, 0.5, 0.4),
            (5, 0.92, 0.16),
            (6, 1, 0),
            (7, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert ptp.q(t) == q
            assert ptp.dq(t) == dq

    def test_update_triangular_velocity_ndarray(self) -> None:
        ptp = PTP(np.array([0, 0, 0]), np.array([1, 1, 1]), 5, profile_name="tri")
        expectation_table = [
            # t, q, dq
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0.08, 0.16),
            (2.5, 0.5, 0.4),
            (4, 0.92, 0.16),
            (5, 1, 0),
            (6, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert_array_equal(ptp.q(t), np.array([q, q, q]))
            assert_array_equal(ptp.dq(t), np.array([dq, dq, dq]))

    def test_update_5th_degree_polynomial(self) -> None:
        ptp = PTP(0, 1, 10, profile_name="5th")
        expectation_table = [
            # t, q, dq
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0.00856, 0.0243),
            (5, 0.5, 0.1875),
            (8, 2944 / 3125, 48 / 625),
            (10, 1, 0),
            (11, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert ptp.q(t) == pytest.approx(q)
            assert ptp.dq(t) == pytest.approx(dq)

    def test_update_5th_degree_polynomial_t0(self) -> None:
        ptp = PTP(0, 1, 10, 1, profile_name="5th")
        expectation_table = [
            # t, q, dq
            (0, 0, 0),
            (1, 0, 0),
            (2, 0.00856, 0.0243),
            (6, 0.5, 0.1875),
            (9, 2944 / 3125, 48 / 625),
            (11, 1, 0),
            (12, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert ptp.q(t) == pytest.approx(q)
            assert ptp.dq(t) == pytest.approx(dq)

    def test_update_5th_degree_polynomial_ndarray(self) -> None:
        ptp = PTP(np.array([0, 0, 0]), np.array([1, 1, 1]), 10, profile_name="5th")
        expectation_table = [
            # t, q, dq
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0.00856, 0.0243),
            (5, 0.5, 0.1875),
            (8, 2944 / 3125, 48 / 625),
            (10, 1, 0),
            (11, 1, 0),
        ]
        for t, q, dq in expectation_table:
            assert_array_almost_equal(ptp.q(t), [q, q, q])
            assert_array_almost_equal(ptp.dq(t), [dq, dq, dq])

    def test_set_vmax(self) -> None:
        ptp = PTP(0, 1, 5, vmax=0.25, profile_name="tra")
        assert ptp.profile.vmax == 0.25  # type: ignore
        assert ptp.profile.tb == 1  # type: ignore

    def test_set_vmax_ndarray(self) -> None:
        ptp = PTP(
            np.array([0, 0]),
            np.array([1, 1]),
            5,
            vmax=np.array([0.25, 0.3]),
            profile_name="tra",
        )
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.3]))  # type: ignore
        assert_array_almost_equal(ptp.profile.tb, np.array([1, 5 / 3]))  # type: ignore

    def test_set_vmax_adapt_dimension_to_q0(self) -> None:
        ptp = PTP(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            5,
            vmax=0.25,
            profile_name="tra",
        )
        assert ptp.profile.vmax.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.25, 0.25]))  # type: ignore
        assert ptp.profile.tb.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.tb, np.array([1, 1, 1]))  # type: ignore

    def test_set_vmax_set_zero(self) -> None:
        ptp = PTP(0, 1, 6, vmax=0, profile_name="tra")
        assert ptp.profile.vmax == 0.25  # type: ignore
        assert ptp.profile.tb == 2  # type: ignore

    def test_set_vmax_set_zero_ndarray(self) -> None:
        ptp = PTP(
            np.array([0, 0]),
            np.array([1, 1]),
            6,
            vmax=np.array([0, 0.3]),
            profile_name="tra",
        )
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.3]))  # type: ignore
        assert_array_almost_equal(ptp.profile.tb, np.array([2, 8 / 3]))  # type: ignore

    def test_set_vmax_set_zero_adapt_dimension_to_q0(self) -> None:
        ptp = PTP(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            6,
            vmax=0,
            profile_name="tra",
        )
        assert ptp.profile.vmax.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.25, 0.25]))  # type: ignore
        assert ptp.profile.tb.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.tb, np.array([2, 2, 2]))  # type: ignore

    def test_set_tb(self) -> None:
        ptp = PTP(0, 1, 5, tb=1, profile_name="tra")
        assert ptp.profile.vmax == 0.25  # type: ignore
        assert ptp.profile.tb == 1  # type: ignore

    def test_set_tb_ndarray(self) -> None:
        ptp = PTP(
            np.array([0, 0]),
            np.array([1, 1]),
            5,
            tb=np.array([1, 5 / 3]),
            profile_name="tra",
        )
        assert_array_almost_equal(ptp.profile.vmax, np.array([0.25, 0.3]))  # type: ignore
        assert_array_equal(ptp.profile.tb, np.array([1, 5 / 3]))  # type: ignore

    def test_set_tb_adapt_dimension_to_q0(self) -> None:
        ptp = PTP(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            5,
            tb=1,
            profile_name="tra",
        )
        assert ptp.profile.vmax.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.25, 0.25]))  # type: ignore
        assert ptp.profile.tb.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.tb, np.array([1, 1, 1]))  # type: ignore

    def test_set_tb_set_zero(self) -> None:
        ptp = PTP(0, 1, 6, tb=0, profile_name="tra")
        assert ptp.profile.vmax == 0.25  # type: ignore
        assert ptp.profile.tb == 2  # type: ignore

    def test_set_tb_set_zero_ndarray(self) -> None:
        ptp = PTP(
            np.array([0, 0]),
            np.array([1, 1]),
            6,
            tb=np.array([0, 8 / 3]),
            profile_name="tra",
        )
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.3]))  # type: ignore
        assert_array_almost_equal(ptp.profile.tb, np.array([2, 8 / 3]))  # type: ignore

    def test_set_tb_set_zero_adapt_dimension_to_q0(self) -> None:
        ptp = PTP(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            6,
            tb=0,
            profile_name="tra",
        )
        assert ptp.profile.vmax.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.vmax, np.array([0.25, 0.25, 0.25]))  # type: ignore
        assert ptp.profile.tb.shape == (3,)  # type: ignore
        assert_array_equal(ptp.profile.tb, np.array([2, 2, 2]))  # type: ignore

    def test_error_no_vmax_tb_provided(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _ = PTP(0, 1, 5, profile_name="tra")
        msg = "Require Vmax or Tb for trapezoidal velocity profile"
        assert msg in str(excinfo.value)

    def test_error_too_small_vmax(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _ = PTP(0, 1, 5, profile_name="tra", vmax=0.1)
        msg = "Specified Vmax for q[0] is too small to reach desired position: 0.1"
        assert msg in str(excinfo.value)

    def test_error_too_small_vmax_ndarray(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _ = PTP(
                np.array([0, 0, 0]),
                np.array([1, 1, 1]),
                5,
                profile_name="tra",
                vmax=np.array([0.25, 0.1, 0.25]),
            )
        msg = "Specified Vmax for q[1] is too small to reach desired position: 0.1"
        assert msg in str(excinfo.value)

    def test_warn_too_large_vmax(self) -> None:
        with pytest.warns(ResourceWarning) as record:
            _ = PTP(0, 1, 5, profile_name="tra", vmax=0.5)
        msg = "Specified Vmax for q[0] is truncated: 0.5 -> 0.4"
        assert msg in str(record[0].message)

    def test_warn_too_large_vmax_ndarray(self) -> None:
        with pytest.warns(ResourceWarning) as record:
            ptp = PTP(
                np.array([0, 0, 0, 0]),
                np.array([1, 1, 1, 1]),
                5,
                profile_name="tra",
                vmax=np.array([0.25, 0.5, 0.25, 0.6]),
            )
        assert len(record) == 2
        msg = "Specified Vmax for q[1] is truncated: 0.5 -> 0.4"
        assert msg in str(record[0].message)
        msg = "Specified Vmax for q[3] is truncated: 0.6 -> 0.4"
        assert msg in str(record[1].message)
        assert_array_equal(ptp.profile.vmax, [0.25, 0.4, 0.25, 0.4])  # type: ignore

    def test_warn_too_large_tb(self) -> None:
        with pytest.warns(ResourceWarning) as record:
            _ = PTP(0, 1, 5, profile_name="tra", tb=3)
        msg = "Specified Tb for q[0] is reduced: 3.0 -> 2.5"
        assert msg in str(record[0].message)

    def test_warn_too_large_tb_ndarray(self) -> None:
        with pytest.warns(ResourceWarning) as record:
            ptp = PTP(
                np.array([0, 0, 0, 0]),
                np.array([1, 1, 1, 1]),
                5,
                profile_name="tra",
                tb=np.array([4, 1, 2, 5]),
            )
        assert len(record) == 2
        msg = "Specified Tb for q[0] is reduced: 4.0 -> 2.5"
        assert msg in str(record[0].message)
        msg = "Specified Tb for q[3] is reduced: 5.0 -> 2.5"
        assert msg in str(record[1].message)
        assert_array_equal(ptp.profile.tb, [2.5, 1, 2, 2.5])  # type: ignore

    def test_update_trapezoidal_velocity(self) -> None:
        ptp = PTP(0, 1, 5, vmax=0.25, profile_name="tra")
        expectation_table = [
            # t, q, dq, ddq
            (-1, 0, 0, 0),
            (0, 0, 0, 0.25),
            (0.5, 0.03125, 0.125, 0.25),
            (1, 0.125, 0.25, 0),
            (2.5, 0.5, 0.25, 0),
            (4, 0.875, 0.25, -0.25),
            (4.5, 0.96875, 0.125, -0.25),
            (5, 1, 0, 0),
            (6, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert ptp.q(t) == q
            assert ptp.dq(t) == dq
            assert ptp.ddq(t) == ddq

    def test_update_trapezoidal_velocity_t0(self) -> None:
        ptp = PTP(0, 1, 5, 1, vmax=0.25, profile_name="tra")
        expectation_table = [
            # t, q, dq
            (0, 0, 0, 0),
            (1, 0, 0, 0.25),
            (1.5, 0.03125, 0.125, 0.25),
            (2, 0.125, 0.25, 0),
            (3.5, 0.5, 0.25, 0),
            (5, 0.875, 0.25, -0.25),
            (5.5, 0.96875, 0.125, -0.25),
            (6, 1, 0, 0),
            (7, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert ptp.q(t) == q
            assert ptp.dq(t) == dq
            assert ptp.ddq(t) == ddq

    def test_update_trapezoidal_velocity_ndarray(self) -> None:
        ptp = PTP(
            np.array([0, 0, 0]), np.array([1, 1, 1]), 5, vmax=0.25, profile_name="tra"
        )
        expectation_table = [
            # t, q, dq, ddq
            (-1, 0, 0, 0),
            (0, 0, 0, 0.25),
            (0.5, 0.03125, 0.125, 0.25),
            (1, 0.125, 0.25, 0),
            (2.5, 0.5, 0.25, 0),
            (4, 0.875, 0.25, -0.25),
            (4.5, 0.96875, 0.125, -0.25),
            (5, 1, 0, 0),
            (6, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert_array_almost_equal(ptp.q(t), [q, q, q])
            assert_array_almost_equal(ptp.dq(t), [dq, dq, dq])
            assert_array_almost_equal(ptp.ddq(t), [ddq, ddq, ddq])

    def test_update_trapezoidal_velocity_ndarray_specify_vmax(self) -> None:
        ptp = PTP(
            np.array([0, 0]),
            np.array([1, 1]),
            5,
            vmax=np.array([0.25, 0.4]),
            profile_name="tra",
        )
        expectation_table = [
            # t, q, dq, ddq
            (-1, (0, 0), (0, 0), (0, 0)),
            (0, (0, 0), (0, 0), (0.25, 0.16)),
            (0.5, (0.03125, 0.02), (0.125, 0.08), (0.25, 0.16)),
            (1, (0.125, 0.08), (0.25, 0.16), (0, 0.16)),
            (2.5, (0.5, 0.5), (0.25, 0.4), (0, -0.16)),
            (4, (0.875, 0.92), (0.25, 0.16), (-0.25, -0.16)),
            (4.5, (0.96875, 0.98), (0.125, 0.08), (-0.25, -0.16)),
            (5, (1, 1), (0, 0), (0, 0)),
            (6, (1, 1), (0, 0), (0, 0)),
        ]
        for t, q, dq, ddq in expectation_table:
            assert_array_almost_equal(ptp.q(t), q)
            assert_array_almost_equal(ptp.dq(t), dq)
            assert_array_almost_equal(ptp.ddq(t), ddq)

    def test_update_sinusoidal_velocity(self) -> None:
        ptp = PTP(0, 1, 4, profile_name="sin")
        expectation_table = [
            # t, q, dq, ddq
            (-1, 0, 0, 0),
            (0, 0, 0, 0),
            (1, 0.25 - 1.0 / PIx2, 0.25, PIx2 / 16),
            (2, 0.5, 0.5, 0),
            (3, 0.75 + 1.0 / PIx2, 0.25, -PIx2 / 16),
            (4, 1, 0, 0),
            (5, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert ptp.q(t) == pytest.approx(q)
            assert ptp.dq(t) == pytest.approx(dq)
            assert ptp.ddq(t) == pytest.approx(ddq)

    def test_update_sinusoidal_velocity_t0(self) -> None:
        ptp = PTP(0, 1, 4, 1, profile_name="sin")
        expectation_table = [
            # t, q, dq, ddq
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (2, 0.25 - 1.0 / PIx2, 0.25, PIx2 / 16),
            (3, 0.5, 0.5, 0),
            (4, 0.75 + 1.0 / PIx2, 0.25, -PIx2 / 16),
            (5, 1, 0, 0),
            (6, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert ptp.q(t) == pytest.approx(q)
            assert ptp.dq(t) == pytest.approx(dq)
            assert ptp.ddq(t) == pytest.approx(ddq)

    def test_update_sinusoidal_velocity_ndarray(self) -> None:
        ptp = PTP(np.array([0, 0, 0]), np.array([1, 1, 1]), 4, profile_name="sin")
        expectation_table = [
            # t, q, dq, ddq
            (-1, 0, 0, 0),
            (0, 0, 0, 0),
            (1, 0.25 - 1.0 / PIx2, 0.25, PIx2 / 16),
            (2, 0.5, 0.5, 0),
            (3, 0.75 + 1.0 / PIx2, 0.25, -PIx2 / 16),
            (4, 1, 0, 0),
            (5, 1, 0, 0),
        ]
        for t, q, dq, ddq in expectation_table:
            assert_array_almost_equal(ptp.q(t), [q, q, q])
            assert_array_almost_equal(ptp.dq(t), [dq, dq, dq])
            assert_array_almost_equal(ptp.ddq(t), [ddq, ddq, ddq])
