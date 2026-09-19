"""What the browser is allowed to ask the solver for, ddsim/api/sweeps.py.

The same argument api/devices.py makes about device constructors, made about
the three sweeps and the transport models: the knobs offered are read from the
functions' own signatures, so a default changed in extract/iv.py is the default
the browser shows and a knob added there is offered the moment it exists.

One thing here is load-bearing beyond convenience. TransportModels.for_device
defaults to constant mobility with no field dependence and no surface
scattering, because that is what every result before Phase 5 was taken with.
A MOSFET solved that way has no velocity saturation in it. The API neither
hides that nor invents a second set of defaults: it hands the browser
for_device's own defaults to render, and the person pressing solve chooses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ddsim.api.devices import build_from_spec
from ddsim.api.sweeps import (
    SWEEP_KINDS,
    build_models,
    check_request,
    model_parameters,
    run_sweep,
    sweep_parameters,
)
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import MOBILITY_MODELS, TransportModels
from ddsim.extract.cv import CVCurve, CVFrame, Response
from ddsim.extract.iv import IVCurve, IVFrame

DIODE = {"n_nodes": 61, "h_min": 5e-7}
"""A coarse diode, built through the API the way the browser builds one."""

CAP: dict[str, Any] = {}
"""The default capacitor."""

FET = {
    "n_contact": 4,
    "n_sd": 10,
    "n_channel": 12,
    "n_silicon": 29,
    "n_oxide": 4,
    "h_min_x": 5e-7,
    "h_min_y": 1e-7,
    "drain_voltage": 0.05,
}
"""The coarse MOSFET the unit tests share."""


def diode():
    return build_from_spec("pn_diode", DIODE)


def named(parameters) -> dict[str, Any]:
    return {p.name: p for p in parameters}


# ------------------------------------------------------------ what is offered


def test_the_three_sweeps_the_phase_names_are_the_ones_offered() -> None:
    """A diode I-V, a MOS C-V and a MOSFET transfer curve. Anything else the
    browser could ask for does not exist yet."""
    assert set(SWEEP_KINDS) == {"iv", "transfer", "cv"}


def test_an_unknown_sweep_is_refused_and_the_known_ones_are_listed() -> None:
    with pytest.raises(ValueError, match="unknown sweep") as raised:
        sweep_parameters("iv_curve")
    for kind in SWEEP_KINDS:
        assert kind in str(raised.value)


@pytest.mark.parametrize("kind", sorted(SWEEP_KINDS))
def test_a_sweep_offers_its_own_numeric_knobs(kind) -> None:
    """Read from the signature, so this is a list nobody maintains."""
    offered = named(sweep_parameters(kind))

    assert "max_iterations" in offered
    assert offered["max_iterations"].type == "int"


def test_the_continuation_knobs_come_with_the_sweeps_own_defaults() -> None:
    """iv_sweep declares step=0.05. A browser showing anything else would be
    showing a sweep the CLI does not run."""
    offered = named(sweep_parameters("iv"))

    assert offered["step"].default == 0.05
    assert offered["start"].default == 0.0


@pytest.mark.parametrize("kind", sorted(SWEEP_KINDS))
def test_nothing_that_is_not_a_json_scalar_is_offered(kind) -> None:
    """The device, the voltage list, the contact names and the telemetry
    callback are the request's own business and are not knobs on a form. A
    client that could set on_frame is a client that could ask for a callback
    the server has no way to honour."""
    offered = named(sweep_parameters(kind))

    for hidden in ("device", "voltages", "contact", "models", "on_frame"):
        assert hidden not in offered


def test_the_model_flags_carry_for_devices_own_defaults() -> None:
    """The one place in the project these defaults are written down is
    TransportModels.for_device, and this is read from it."""
    offered = named(model_parameters())

    assert offered["mobility"].default == "constant"
    assert offered["mobility"].choices == MOBILITY_MODELS
    assert offered["field_dependent"].default is False
    assert offered["surface"].default is False
    assert offered["auger"].default is False


def test_the_model_flags_do_not_offer_an_object_the_browser_cannot_build() -> None:
    """recombination is a model instance, not a number. Offering it would
    invite a device running physics this project never validated."""
    assert "recombination" not in named(model_parameters())


# ----------------------------------------------------------- what is refused


def test_a_knob_the_sweep_does_not_have_is_refused() -> None:
    """Refused rather than dropped. A dropped typo runs the default sweep and
    plots it as though it were the one that was asked for."""
    with pytest.raises(ValueError, match="stepsize"):
        run_sweep("iv", diode(), "anode", [0.1], settings={"stepsize": 0.2})


def test_a_knob_of_the_wrong_type_is_refused() -> None:
    """max_iterations is a count. 2.5 cycles is not a budget."""
    with pytest.raises(TypeError, match="max_iterations"):
        run_sweep("iv", diode(), "anode", [0.1], settings={"max_iterations": 2.5})


def test_an_unknown_mobility_model_is_refused_by_the_solver_itself() -> None:
    """Not by a second list here. The names live in transport.MOBILITY_MODELS
    and the refusal is the solver's own."""
    with pytest.raises(ValueError, match="unknown mobility model"):
        build_models(diode(), {"mobility": "bogus"})


def test_a_model_name_that_is_not_a_name_is_refused() -> None:
    """mobility is a name, not a number. A 5 arriving here means the form
    sent the wrong field, and running the default model instead would be a
    curve labelled with physics it was not taken with."""
    with pytest.raises(TypeError, match="mobility"):
        build_models(diode(), {"mobility": 5})


def test_a_model_flag_that_is_not_a_boolean_is_refused() -> None:
    """A truthy 1 is not a flag. Accepting it here is how a sweep ends up
    field dependent because a checkbox sent the wrong shape."""
    with pytest.raises(TypeError, match="field_dependent"):
        build_models(diode(), {"field_dependent": 1})


def test_a_capacitance_sweep_refuses_transport_models() -> None:
    """A C-V point is an equilibrium Poisson solve. There is no mobility in it
    at all, so accepting a mobility and ignoring it would report a curve that
    the flags on screen do not describe."""
    with pytest.raises(ValueError, match="cv"):
        run_sweep(
            "cv",
            build_from_spec("mos_cap", CAP),
            "gate",
            [-1.0],
            models={"mobility": "arora"},
        )


def test_measuring_at_another_terminal_is_refused_where_it_means_nothing() -> None:
    """Only a transfer curve sweeps one terminal and measures another."""
    with pytest.raises(ValueError, match="measure_at"):
        run_sweep("iv", diode(), "anode", [0.1], measure_at="cathode")


def test_a_transfer_sweep_checks_the_drain_it_measures_by_default() -> None:
    """A transfer curve with no measure_at reads the drain. A MOS capacitor
    has no drain, and the check that runs before a job is started let that
    through because it only looked at a measure_at that was named, so the job
    started and died on a KeyError a second later."""
    with pytest.raises(KeyError, match="drain"):
        check_request("transfer", build_from_spec("mos_cap", CAP), "gate")



# -------------------------------------------------------------- what it runs


def test_a_diode_sweep_comes_back_as_an_iv_curve() -> None:
    curve, _ = run_sweep("iv", diode(), "anode", [0.0, 0.2])

    assert isinstance(curve, IVCurve)
    assert curve.complete
    assert curve.voltage.tolist() == [0.0, 0.2]


def test_a_capacitance_sweep_comes_back_as_a_cv_curve() -> None:
    curve, _ = run_sweep("cv", build_from_spec("mos_cap", CAP), "gate", [-1.0, 0.0])

    assert isinstance(curve, CVCurve)
    assert curve.complete


def test_a_capacitance_sweep_takes_the_response_it_is_given() -> None:
    """The high frequency approximation is a different curve, not a different
    presentation of the same one."""
    curve, _ = run_sweep(
        "cv",
        build_from_spec("mos_cap", CAP),
        "gate",
        [1.0],
        settings={"response": "high_frequency"},
    )

    assert curve.response is Response.HIGH_FREQUENCY


def test_a_transfer_curve_measures_the_drain_by_default() -> None:
    """No current flows in a gate, so a gate current curve is flat at zero."""
    curve, _ = run_sweep(
        "transfer",
        build_from_spec("nmos", FET),
        "gate",
        [0.2, 0.4],
        settings={"step": 0.2},
    )

    assert curve.measured_at == "drain"
    assert curve.complete


def test_the_models_the_flags_ask_for_are_the_models_that_are_built() -> None:
    """The flag that matters most: without field_dependent there is no
    velocity saturation anywhere in the solve."""
    plain = build_models(diode(), None)
    saturating = build_models(diode(), {"field_dependent": True})

    assert not plain.field_dependent
    assert saturating.field_dependent


def test_arora_makes_the_diffusivity_vary_along_the_device() -> None:
    """A doping dependent mobility is an array over edges. A scalar coming
    back would mean the flag was accepted and then ignored."""
    constant = build_models(diode(), {"mobility": "constant"})
    arora = build_models(diode(), {"mobility": "arora"})

    assert np.asarray(constant.Dn).ndim == 0
    assert np.asarray(arora.Dn).ndim == 1


def test_a_sweep_hands_back_the_models_it_solved_with() -> None:
    device = pn_diode(n_nodes=61, h_min=5e-7)
    _, models = run_sweep("iv", device, "anode", [0.0, 0.1])

    assert isinstance(models, TransportModels)


def test_a_capacitance_sweep_has_no_transport_models() -> None:
    device = mos_cap()
    _, models = run_sweep("cv", device, "gate", [0.0])

    assert models is None


# ----------------------------------------------------------------- telemetry


@pytest.mark.parametrize(
    ("kind", "device", "contact", "voltages", "settings", "frame"),
    [
        ("iv", diode, "anode", [0.1], None, IVFrame),
        (
            "transfer",
            lambda: build_from_spec("nmos", FET),
            "gate",
            [0.2],
            {"step": 0.2},
            IVFrame,
        ),
        (
            "cv",
            lambda: build_from_spec("mos_cap", CAP),
            "gate",
            [-1.0],
            None,
            CVFrame,
        ),
    ],
    ids=["iv", "transfer", "cv"],
)
def test_every_sweep_reports_through_the_callback(
    kind, device, contact, voltages, settings, frame
) -> None:
    """The whole point of the layer. A sweep the API can run but not watch
    would be a page with a spinner on it."""
    frames: list[Any] = []

    run_sweep(
        kind,
        device(),
        contact,
        voltages,
        settings=settings,
        on_frame=frames.append,
    )

    assert any(isinstance(sent, frame) for sent in frames)
