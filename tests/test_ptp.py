import pytest
from affctrllib.ptp import PTP, FifthOrderPolynomialProfile, TriangularVelocityProfile


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
            ("5th-order polynomial", FifthOrderPolynomialProfile),
            ("5th order polynomial", FifthOrderPolynomialProfile),
            ("5th-order", FifthOrderPolynomialProfile),
            ("5th order", FifthOrderPolynomialProfile),
            ("5th", FifthOrderPolynomialProfile),
        ],
    )
    def test_select_profile(self, name, expected) -> None:
        ptp = PTP(0, 1, 5, profile_name=name)
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

    def test_update_5th_order_polynomial(self) -> None:
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

    @pytest.mark.skip
    def test_update_5th_order_polynomial_t0(self) -> None:
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
            assert ptp.q(t) == q
            assert ptp.dq(t) == dq