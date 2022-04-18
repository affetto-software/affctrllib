import os

import numpy as np
import pytest
from affctrllib.affctrl import AffCtrl, FeedbackPID, FeedbackPIDF
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


class TestFeedbackPID:
    def test_init(self) -> None:
        _ = FeedbackPID()
        assert True

    @pytest.mark.parametrize("kP", [50, [10, 20], np.array([10, 20, 30])])
    def test_init_with_kP(self, kP) -> None:
        pid = FeedbackPID(kP=kP)
        if isinstance(kP, np.ndarray):
            assert_array_equal(pid.kP, kP)
        else:
            assert pid.kP == kP

    @pytest.mark.parametrize("kD", [500, [100, 200], np.array([100, 200, 300])])
    def test_init_with_kD(self, kD) -> None:
        pid = FeedbackPID(kD=kD)
        if isinstance(kD, np.ndarray):
            assert_array_equal(pid.kD, kD)
        else:
            assert pid.kD == kD

    @pytest.mark.parametrize("kI", [5, [1, 2], np.array([1, 2, 3])])
    def test_init_with_kI(self, kI) -> None:
        pid = FeedbackPID(kI=kI)
        if isinstance(kI, np.ndarray):
            assert_array_equal(pid.kI, kI)
        else:
            assert pid.kI == kI

    @pytest.mark.parametrize("stiff", [127, [150, 150], np.array([200, 200, 200])])
    def test_init_with_stiff(self, stiff) -> None:
        pid = FeedbackPID(stiff=stiff)
        if isinstance(stiff, np.ndarray):
            assert_array_equal(pid.stiff, stiff)
        else:
            assert pid.stiff == stiff

    @pytest.mark.parametrize("kP", [50, [10, 20], np.array([10, 20, 30])])
    def test_kP_setter(self, kP) -> None:
        pid = FeedbackPID()
        pid.kP = kP
        if isinstance(kP, np.ndarray):
            assert_array_equal(pid.kP, kP)
        else:
            assert pid.kP == kP

    @pytest.mark.parametrize("kD", [500, [100, 200], np.array([100, 200, 300])])
    def test_kD_setter(self, kD) -> None:
        pid = FeedbackPID()
        pid.kD = kD
        if isinstance(kD, np.ndarray):
            assert_array_equal(pid.kD, kD)
        else:
            assert pid.kD == kD

    @pytest.mark.parametrize("kI", [5, [1, 2], np.array([1, 2, 3])])
    def test_kI_setter(self, kI) -> None:
        pid = FeedbackPID()
        pid.kI = kI
        if isinstance(kI, np.ndarray):
            assert_array_equal(pid.kI, kI)
        else:
            assert pid.kI == kI

    @pytest.mark.parametrize("stiff", [127, [150, 150], np.array([200, 200, 200])])
    def test_stiff_setter(self, stiff) -> None:
        pid = FeedbackPID()
        pid.stiff = stiff
        if isinstance(stiff, np.ndarray):
            assert_array_equal(pid.stiff, stiff)
        else:
            assert pid.stiff == stiff

    @pytest.mark.parametrize(
        "t,q,dq,qdes,dqdes,expected",
        [
            (0, 0, 0, 5, 0, 5),
            (1, 2, 4, 5, 0, 3),
            (2, 4, 2, 5, 0, 1),
            (3, 6, -2, 5, 0, -1),
            (4, 8, -4, 5, 0, -3),
        ],
    )
    def test_positional_feedback_kP(self, t, q, dq, qdes, dqdes, expected) -> None:
        pid = FeedbackPID(kP=1, kD=0, kI=0)
        u = pid.positional_feedback(t, q, dq, qdes, dqdes)
        assert u == expected

    @pytest.mark.parametrize(
        "t,q,dq,qdes,dqdes,expected",
        [
            (0, 0, 0, 5, 0, 5),
            (1, 2, 4, 5, 0, -1),
            (2, 4, 2, 5, 0, -1),
            (3, 6, -2, 5, 0, 1),
            (4, 8, -4, 5, 0, 1),
        ],
    )
    def test_positional_feedback_kP_kD(self, t, q, dq, qdes, dqdes, expected) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=0)
        u = pid.positional_feedback(t, q, dq, qdes, dqdes)
        assert u == expected

    def test_positional_feedback_kP_kD_kI(self) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=1)
        q = [0, 2, 4, 6, 8]
        dq = [0, 4, 2, -2, -4]
        qdes = [5, 5, 5, 5, 5]
        dqdes = [0, 0, 0, 0, 0]
        expected = [10, 7, 8, 9, 6]
        for t in range(5):
            u = pid.positional_feedback(t, q[t], dq[t], qdes[t], dqdes[t])
            assert u == expected[t]

    def test_update(self) -> None:
        pid = FeedbackPID(kP=1, kD=1, kI=1, stiff=2)
        q = [0, 2, 4, 6, 8]
        dq = [0, 4, 2, -2, -4]
        qdes = [5, 5, 5, 5, 5]
        dqdes = [0, 0, 0, 0, 0]
        expected = [(12, -8), (9, -5), (10, -6), (11, -7), (8, -4)]
        for t in range(5):
            ua, ub = pid.update(t, 0, 0, q[t], dq[t], qdes[t], dqdes[t])
            assert ua, ub == expected[t]


class TestAffCtrl:
    def test_init(self) -> None:
        ctrl = AffCtrl()
        assert isinstance(ctrl, AffCtrl)

    def test_init_config(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "default.toml")
        ctrl = AffCtrl(config)
        assert str(ctrl.config_path) == config
        assert ctrl.dof == 13
        assert isinstance(ctrl.feedback_scheme, FeedbackPID)
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([20] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([200] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([2] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([150] * 13))

    def test_init_config_alternative(self) -> None:
        config = os.path.join(CONFIG_DIR_PATH, "alternative.toml")
        ctrl = AffCtrl(config)
        assert str(ctrl.config_path) == config
        assert ctrl.dof == 14
        assert isinstance(ctrl.feedback_scheme, FeedbackPIDF)
        assert_array_equal(ctrl.feedback_scheme.kP, np.array([3] * 13))
        assert_array_equal(ctrl.feedback_scheme.kD, np.array([30] * 13))
        assert_array_equal(ctrl.feedback_scheme.kI, np.array([0.3] * 13))
        assert_array_equal(ctrl.feedback_scheme.stiff, np.array([180] * 13))
