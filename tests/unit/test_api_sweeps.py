from __future__ import annotations
from typing import Any
import numpy as np, pytest
from ddsim.api.devices import build_from_spec
from ddsim.api.sweeps import (SWEEP_KINDS, build_models, check_request, model_parameters, run_sweep, sweep_parameters,)
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import MOBILITY_MODELS, TransportModels
from ddsim.extract.ac import ACCurve, ACFrame
from ddsim.extract.cv import CVCurve, CVFrame, Response
from ddsim.extract.iv import IVCurve, IVFrame
DIODE = {'n_nodes': 61, 'h_min': 5e-7}


CAP: dict[str, Any] = {}

SMALL_CAP = {'n_silicon': 41, 'n_oxide': 3, 'h_min': 2e-7}

FET = {
    'n_contact':4,
    'n_sd': 10,
    "n_channel":12,
    'n_silicon': 29,
    'n_oxide':4,
    "h_min_x":5e-7,
    'h_min_y':1e-7,
    "drain_voltage":0.05,
}


def diode():
    return build_from_spec('pn_diode', DIODE)


def named(parameters) -> dict[str, Any]:
    return {p.name: p for p in parameters}


def test_the_sweeps_the_phases_name_are_the_ones_offered() -> None:
    assert set(SWEEP_KINDS) == {'iv', 'transfer', "cv", "ac", "transient"}


def test_an_unknown_sweep_is_refused_and_the_known_ones_are_listed() -> None:
    with pytest.raises(ValueError, match="unknown sweep") as ok:
        sweep_parameters("iv_curve")

    for i in SWEEP_KINDS:
        assert i in str(ok.value)


@pytest.mark.parametrize('kind', sorted(SWEEP_KINDS))
def test_a_sweep_offers_its_own_numeric_knobs(kind) -> None:
    k = named(sweep_parameters(kind))
    assert 'max_iterations' in k
    assert k['max_iterations'].type == 'int'


def test_the_continuation_knobs_come_with_the_sweeps_own_defaults() -> None:
    z = named(sweep_parameters("iv"))
    assert z["step"].default == 0.05
    assert z['start'].default == 0.0

@pytest.mark.parametrize("kind", sorted(SWEEP_KINDS))
def test_nothing_that_is_not_a_json_scalar_is_offered(kind) -> None:
    k = named(sweep_parameters(kind))

    for y in ('device', "voltages", "frequencies", "times", "contact", 'models', "on_frame"):
        assert y not in k

def test_the_model_flags_carry_for_devices_own_defaults() -> None:
    r = named(model_parameters())

    assert r["mobility"].default == 'constant'
    assert r['mobility'].choices == MOBILITY_MODELS
    assert r['field_dependent'].default is False
    assert r["surface"].default is False
    assert r["auger"].default is False


def test_the_model_flags_do_not_offer_an_object_the_browser_cannot_build() -> None:
    assert 'recombination' not in named(model_parameters())


def test_a_knob_the_sweep_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="stepsize"):
        run_sweep('iv', diode(), 'anode', [0.1], settings={"stepsize":0.2})


def test_a_knob_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(TypeError, match='max_iterations'):
        run_sweep("iv", diode(), 'anode', [0.1], settings={"max_iterations": 2.5})


def test_an_unknown_mobility_model_is_refused_by_the_solver_itself() -> None:
    with pytest.raises(ValueError, match="unknown mobility model"):
        build_models(diode(), {"mobility": "bogus"})


def test_a_model_name_that_is_not_a_name_is_refused() -> None:
    with pytest.raises(TypeError, match='mobility'):
        build_models(diode(), {'mobility': 5})
def test_a_model_flag_that_is_not_a_boolean_is_refused() -> None:
    with pytest.raises(TypeError,
        match="field_dependent"):
        build_models(diode(), {"field_dependent": 1})
def test_a_capacitance_sweep_refuses_transport_models() -> None:
    with pytest.raises(ValueError, match="cv"):
        run_sweep(
            "cv",
            build_from_spec("mos_cap", CAP),
            'gate',
            [-1.0],
            models={"mobility":"arora"},
        )


def test_measuring_at_another_terminal_is_refused_where_it_means_nothing() -> None:
    with pytest.raises(ValueError, match='measure_at'):
        run_sweep("iv", diode(), 'anode', [0.1], measure_at='cathode')


def test_a_transfer_sweep_checks_the_drain_it_measures_by_default() -> None:
    with pytest.raises(KeyError,
               match='drain'):
        check_request("transfer", build_from_spec("mos_cap", CAP), 'gate')


def test_a_model_the_device_cannot_take_is_refused_before_the_job() -> None:
    with pytest.raises(TypeError, match='surface mobility'):
        check_request("iv", diode(), "anode", models={'surface': True})

def test_an_iv_refusal_on_a_gated_device_says_what_to_use_instead() -> None:
    with pytest.raises(TypeError,
            match='transfer'):
        check_request("iv", build_from_spec('mos_cap', CAP), "body")


def test_a_diode_sweep_comes_back_as_an_iv_curve() -> None:
    s, _ = run_sweep("iv", diode(), "anode", [0.0, 0.2])

    assert isinstance(s, IVCurve)
    assert s.complete
    assert s.voltage.tolist() == [0.0, 0.2]


def test_a_capacitance_sweep_comes_back_as_a_cv_curve() -> None:
    yy, _ = run_sweep('cv', build_from_spec('mos_cap', CAP), "gate", [-1.0, 0.0])
    assert isinstance(yy, CVCurve)

    assert yy.complete


def test_a_capacitance_sweep_takes_the_response_it_is_given() -> None:
    ii, _ = run_sweep(
        'cv',
        build_from_spec("mos_cap", CAP),
        "gate",
        [1.0],
        settings={'response': 'high_frequency'},
    )
    assert ii.response is Response.HIGH_FREQUENCY


def test_a_transfer_curve_measures_the_drain_by_default() -> None:
    a, _ = run_sweep("transfer", build_from_spec("nmos", FET), "gate", [0.2, 0.4], settings={"step":0.2},)

    assert a.measured_at == "drain"

    assert a.complete


def test_the_models_the_flags_ask_for_are_the_models_that_are_built() -> None:
    bb = build_models(diode(), None)
    r = build_models(diode(), {'field_dependent':True})

    assert not bb.field_dependent
    assert r.field_dependent
def test_arora_makes_the_diffusivity_vary_along_the_device() -> None:
    c2 = build_models(diode(), {'mobility': "constant"})
    tmp2 = build_models(diode(), {"mobility": "arora"})
    assert np.asarray(c2.Dn).ndim == 0
    assert np.asarray(tmp2.Dn).ndim == 1


def test_a_sweep_hands_back_the_models_it_solved_with() -> None:
    j = pn_diode(n_nodes=61, h_min=5e-7)
    _, w = run_sweep("iv", j, "anode", [0.0, 0.1])

    assert isinstance(w, TransportModels)


def test_a_capacitance_sweep_has_no_transport_models() -> None:
    a2 = mos_cap()

    _, bar = run_sweep('cv', a2, "gate", [0.0])

    assert bar is None


@pytest.mark.parametrize(("kind", "device", "contact", "voltages", 'settings', "frame"), [('iv', diode, 'anode', [0.1], None, IVFrame), ("transfer", lambda: build_from_spec("nmos", FET), "gate", [0.2], {'step':0.2}, IVFrame,), ('cv', lambda:build_from_spec('mos_cap', CAP), "gate", [-1.0], None, CVFrame,),], ids=["iv", 'transfer', 'cv'],)
def test_every_sweep_reports_through_the_callback(kind, device, contact, voltages, settings, frame) -> None:
    tmp2: list[Any] = []
    run_sweep(kind, device(), contact, voltages, settings=settings, on_frame=tmp2.append,)
    assert any(isinstance(i, frame) for i in tmp2)


def test_an_ac_sweep_needs_its_frequencies() -> None:
    with pytest.raises(ValueError, match="frequenc"):
        check_request("ac", build_from_spec("mos_cap", SMALL_CAP), "gate")


def test_frequencies_are_refused_on_a_sweep_that_has_none() -> None:
    with pytest.raises(ValueError, match="frequenc"):
        check_request("iv", diode(), "anode", frequencies=[1e3])


@pytest.mark.parametrize("f", [-1.0, float("inf"), float("nan")])
def test_a_frequency_that_is_not_one_is_refused_before_the_job(f) -> None:
    with pytest.raises(ValueError, match="frequenc"):
        check_request("ac", build_from_spec("mos_cap", SMALL_CAP), "gate", frequencies=[1e3, f])


def test_an_ac_sweep_refuses_the_surface_model_it_cannot_linearize() -> None:
    with pytest.raises(ValueError, match="surface"):
        check_request("ac", build_from_spec("nmos", FET), "gate", models={"surface": True}, frequencies=[1e6])


def test_an_ac_sweep_comes_back_as_an_ac_curve_with_its_models() -> None:
    got: list[Any] = []
    c, m = run_sweep("ac", build_from_spec("mos_cap", SMALL_CAP), "gate", [-1.0], frequencies=[1e6], on_frame=got.append)
    assert isinstance(c, ACCurve)
    assert c.complete
    assert c.frequencies == (1e6,)
    assert isinstance(m, TransportModels)
    assert any(isinstance(i, ACFrame) for i in got)


def test_a_step_offers_its_resistor_and_its_tolerance() -> None:
    k = named(sweep_parameters("transient"))
    assert k["resistance"].default == 0.0
    assert k["rtol"].default == 1e-3


def test_a_transient_needs_its_sample_times() -> None:
    with pytest.raises(ValueError, match="time"):
        check_request("transient", diode(), "anode", voltages=[0.0, 0.3])


def test_sample_times_are_refused_on_a_sweep_that_has_none() -> None:
    with pytest.raises(ValueError, match="time"):
        check_request("iv", diode(), "anode", times=[1e-9])


def test_frequencies_are_refused_on_a_transient() -> None:
    with pytest.raises(ValueError, match="frequenc"):
        check_request("transient", diode(), "anode", voltages=[0.0, 0.3], times=[1e-9], frequencies=[1e3])


@pytest.mark.parametrize("voltages", [[0.3], [0.0, 0.1, 0.3]])
def test_a_transient_with_other_than_two_voltages_is_refused_before_the_job(voltages) -> None:
    with pytest.raises(ValueError, match="two voltages"):
        check_request("transient", diode(), "anode", voltages=voltages, times=[1e-9])


def test_a_falling_sample_time_is_refused_before_the_job() -> None:
    with pytest.raises(ValueError, match="time"):
        check_request("transient", diode(), "anode", voltages=[0.0, 0.3], times=[1e-9, 1e-10])


def test_a_transient_refuses_the_surface_model_no_transient_test_covers() -> None:
    with pytest.raises(ValueError, match="surface"):
        check_request("transient", build_from_spec("nmos", FET), "gate", models={"surface": True}, voltages=[0.0, 0.5], times=[1e-9])


def test_a_transient_comes_back_as_a_transient_curve_with_its_models() -> None:
    from ddsim.extract.transient import TransientCurve, TransientFrame
    got: list[Any] = []
    c, m = run_sweep("transient", diode(), "anode", [0.0, 0.3], times=[1e-12, 1e-9], on_frame=got.append)
    assert isinstance(c, TransientCurve)
    assert c.complete
    assert [p.time for p in c.points] == [1e-12, 1e-9]
    assert isinstance(m, TransportModels)
    assert sum(isinstance(i, TransientFrame) for i in got) == c.steps
