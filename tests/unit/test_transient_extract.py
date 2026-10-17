from __future__ import annotations
import pytest
from ddsim.extract.transient import Transient, TransientPoint, recovery_time, storage_time


def waveform(rows):
    return Transient(points=tuple(TransientPoint(time=t, state=None, currents={"anode": i}, voltages={"anode": v}) for t, v, i in rows), complete=True)


RECOVERY = waveform([
    (1.0, 0.7, -1.0),
    (2.0, 0.6, -1.0),
    (3.0, 0.2, -1.0),
    (4.0, -0.2, -0.6),
    (5.0, -5.0, -0.3),
    (6.0, -9.0, -0.05),
])


def test_storage_ends_where_the_voltage_crosses_zero_interpolated():
    assert storage_time(RECOVERY, "anode") == pytest.approx(3.5)


def test_recovery_ends_where_the_current_falls_to_the_fraction_interpolated():
    assert recovery_time(RECOVERY, "anode") == pytest.approx(5.0 + 0.2 / 0.25)
    assert recovery_time(RECOVERY, "anode", fraction=0.5) == pytest.approx(4.0 + 0.1 / 0.3)


def test_a_voltage_that_never_crosses_zero_has_no_storage_time():
    with pytest.raises(ValueError, match="never"):
        storage_time(waveform([(1.0, 0.7, -1.0), (2.0, 0.5, -1.0)]), "anode")


def test_a_current_that_never_falls_far_enough_has_no_recovery_time():
    with pytest.raises(ValueError, match="never"):
        recovery_time(RECOVERY, "anode", fraction=0.01)


def test_an_unknown_contact_is_named():
    with pytest.raises(KeyError, match="gate"):
        storage_time(RECOVERY, "gate")
