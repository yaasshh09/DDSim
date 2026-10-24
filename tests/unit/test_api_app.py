from __future__ import annotations
import json; from typing import Any
import numpy as np, pytest
from fastapi.testclient import TestClient
from ddsim.api.app import create_app
from ddsim.api.frames import decode_fields
from ddsim.api.jobs import JobRegistry, JobStatus
from ddsim.device.mos_cap import mos_cap; from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.ac import ROUNDOFF, ac_sweep
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep, iv_sweep

DIODE = {"n_nodes":61, 'h_min':5e-7}

FET = {'n_contact':4, "n_sd":10, "n_channel": 12, "n_silicon": 29, "n_oxide": 4, "h_min_x": 5e-7, 'h_min_y':1e-7, 'drain_voltage':0.05,}
LONG = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

@pytest.fixture
def client():
    with TestClient(create_app()) as dd:
        yield dd


def diode_request(voltages=(0.0, 0.2), **sweep: Any) -> dict:
    return {"device": {"kind": "pn_diode", "parameters": DIODE}, "sweep": {'kind': "iv", 'contact': "anode", "voltages": list(voltages), **sweep,},}


def submit(client, request: dict) -> str:
    vals = client.post('/api/jobs', json=request)
    assert vals.status_code == 200, vals.text
    return vals.json()['id']
def drain(client, job_id: str) -> list[Any]:
    v:list[Any] = []
    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as stuff:
        while True:
            lst = json.loads(stuff.receive_text())
            v.append(lst)
            if lst["type"] == "status":
                return v


def test_the_schema_offers_the_devices_the_registry_knows(client) -> None:
    f = client.get('/api/schema').json()

    assert set(f["devices"]) == {'pn_diode', "mos_cap", 'nmos', 'stack', "drawing"}
    assert set(f['sweeps']) == {"iv", 'transfer', "cv", "ac", "transient"}


def test_the_schema_carries_the_constructors_own_defaults(client) -> None:
    vals = client.get('/api/schema').json()
    d = {p['name']: p for p in vals["devices"]["pn_diode"]}

    assert d["Na"]["default"] == pytest.approx(1e16)
    assert d["n_nodes"]["type"] == 'int'


def test_the_schema_carries_the_model_flags_and_their_choices(client) -> None:
    b2 = client.get('/api/schema').json(); res2 = {p['name']:p for p in b2["models"]}

    assert res2["mobility"]['choices'] == ['constant', "arora"]
    assert res2['field_dependent']['default'] is False


def test_the_schema_carries_the_coarse_presets_and_their_notes(client) -> None:
    z = client.get('/api/schema').json()["presets"]
    assert set(z) == {"mos_cap", "nmos", "drawing"}; assert '1.5 mV' in z['drawing']['note']
    assert z["nmos"]['parameters']["n_silicon"] == 29
    assert '0.621 percent' in z['nmos']['note']


def test_an_unknown_device_is_refused_with_the_known_ones_named(client) -> None:
    z = client.post(
        "/api/jobs",
        json={
            'device': {"kind": "bjt", 'parameters': {}},
            "sweep":{"kind":'iv', 'contact': "anode", "voltages": [0.1]},
        },
    )
    assert z.status_code == 400
    assert "pn_diode" in z.json()["detail"]


def test_a_parameter_of_the_wrong_type_is_refused(client) -> None:
    z = client.post("/api/jobs", json={"device": {'kind': "pn_diode", "parameters": {'n_nodes': 61.5}}, 'sweep': {"kind": "iv", "contact": 'anode', "voltages": [0.1]},},)

    assert z.status_code == 400
    assert "n_nodes" in z.json()["detail"]


def test_a_contact_the_device_does_not_have_is_refused(client) -> None:
    num = client.post("/api/jobs", json={'device': {"kind": "pn_diode", "parameters":DIODE}, "sweep":{"kind":'iv', "contact":"gate", 'voltages':[0.1]},},)

    assert num.status_code == 400
    assert "gate" in num.json()['detail']
    assert num.json()['detail'].startswith("no contact named 'gate' that the iv")


def test_a_request_missing_its_sweep_is_refused_by_the_schema(client) -> None:
    bb = client.post('/api/jobs', json={"device":{'kind': 'pn_diode'}})

    assert bb.status_code == 422
def test_an_unknown_job_is_not_found(client) -> None:
    assert client.get('/api/jobs/nosuchjob').status_code == 404

def test_a_stream_carries_solver_frames_and_then_a_status(client) -> None:
    i = submit(client,
                      diode_request())
    m = drain(client, i)

    k = {flag["type"] for flag in m}
    assert "gummel" in k

    assert "point" in k
    assert m[-1]["type"] == "status"
    assert m[-1]["status"] == JobStatus.DONE.value
def test_telemetry_arrives_before_the_job_is_finished(client) -> None:
    v2 = submit(client,
               diode_request(voltages=LONG))

    with client.websocket_connect(f"/api/jobs/{v2}/stream") as t:
        ss = json.loads(t.receive_text())
        u = client.get(f"/api/jobs/{v2}").json()["status"]
    assert ss["type"] != 'status'
    assert u == JobStatus.RUNNING.value


def test_a_stalled_sweep_is_reported_as_the_measurement_it_is(client) -> None:
    r = submit(
        client,
        diode_request(
            voltages=(0.2, 0.4, 5.0),
            settings={"step":0.2, "max_iterations": 8},
        ),
    )

    b = drain(client, r)
    assert b[-1]["status"] == JobStatus.DONE.value
    y = client.get(f"/api/jobs/{r}/result").json()
    assert y['complete'] is False
    assert y['message']


def test_a_device_that_cannot_be_built_is_refused_before_it_is_a_job(client,) -> None:
    j = client.post("/api/jobs", json={'device': {"kind": 'pn_diode', 'parameters':{'n_nodes': 5}}, "sweep": {"kind": 'iv', 'contact':'anode', 'voltages': [0.1]},},)

    assert j.status_code == 400
    assert "max_ratio" in j.json()['detail']
def test_a_failing_solve_says_why_rather_than_going_quiet(client) -> None:
    y2 = submit(
        client,
        diode_request(
            voltages=(5.2,), settings={"start":5.0, 'max_iterations':1}
        ),
    )

    y = drain(client, y2)
    assert y[-1]['status'] == JobStatus.FAILED.value
    assert 'could not be started' in y[-1]['message']


def test_cancelling_stops_the_solve(client) -> None:
    rr = submit(client, diode_request(voltages=LONG))
    with client.websocket_connect(f"/api/jobs/{rr}/stream") as xx:
        json.loads(xx.receive_text())
        assert client.post(f"/api/jobs/{rr}/cancel").json()["cancelled"]
        while True:
            g = json.loads(xx.receive_text())
            if g["type"] == 'status':
                break

    assert g["status"] == JobStatus.CANCELLED.value
    assert client.get(f"/api/jobs/{rr}").json()['status'] == (
        JobStatus.CANCELLED.value
    )


def test_cancelling_a_finished_job_says_it_changed_nothing(client) -> None:
    x = submit(client, diode_request(voltages=(0.1,)))
    drain(client, x)
    assert client.post(f"/api/jobs/{x}/cancel").json()['cancelled'] is False

def test_the_result_is_available_after_the_stream_has_closed(client) -> None:
    val2 = submit(client, diode_request()); drain(client, val2)

    z = client.get(f"/api/jobs/{val2}/result").json()
    assert z["kind"] == 'iv'
    assert [s['voltage'] for s in z['points']] == [0.0, 0.2]


def test_a_result_asked_for_too_early_is_refused(client) -> None:
    x = submit(client, diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{x}/result").status_code == 409

@pytest.mark.parametrize(("request_body", "direct"), [(diode_request(voltages=(0.0, 0.2)), lambda: iv_sweep(pn_diode(**DIODE), 'anode', [0.0, 0.2], models=TransportModels.for_device(pn_diode(**DIODE)),),), ({'device': {'kind': "mos_cap", "parameters": {}}, 'sweep':{'kind': 'cv', "contact": "gate", 'voltages': [-1.0, 0.0, 1.0],},}, lambda: cv_sweep(mos_cap(), 'gate', [-1.0, 0.0, 1.0]),), ({'device': {'kind': "nmos", 'parameters': FET}, "sweep": {"kind": 'transfer', "contact": 'gate', "voltages": [0.2, 0.4], "settings":{'step': 0.2},},}, lambda:gate_sweep(nmos(**FET), [0.2, 0.4], step=0.2, models=TransportModels.for_device(nmos(**FET)),),),], ids=['diode', "capacitor", 'mosfet'],)
def test_the_browser_gets_bit_for_bit_what_pytest_gets(
    client, request_body, direct
) -> None:
    v = submit(client, request_body)

    drain(client, v)
    w = client.get(f"/api/jobs/{v}/result").json()

    prev = direct()

    a = w['kind'] == 'cv'
    d = "capacitance" if a else "current"

    assert [z['voltage'] for z in w["points"]] == list(
        prev.gate_voltage if a else prev.voltage
    )
    assert [z[d] for z in w["points"]] == list(prev.capacitance if a else prev.current)

def test_the_fields_of_a_point_come_back_as_float32_behind_a_header(
    client,
) -> None:
    i = submit(client, diode_request())
    drain(client, i)

    r = client.get(f"/api/jobs/{i}/fields/1")
    assert r.status_code == 200
    c = decode_fields(r.content)
    assert list(c) == ["x", "psi", "n", 'p', "Ec", "Ev", "Efn", "Efp", "Jx", 'Jy']
    assert np.max(c["n"]) > 1e15


def test_the_fields_of_a_point_know_the_bias_it_was_swept_to(client) -> None:
    a = submit(client, diode_request())
    drain(client, a)

    d2 = decode_fields(client.get(f"/api/jobs/{a}/fields/0").content)
    dd = decode_fields(client.get(f"/api/jobs/{a}/fields/1").content)

    assert not np.any(d2["Jx"])
    assert np.max(np.abs(dd['Jx'])) > 0.0


def test_the_fields_of_a_capacitance_point_carry_its_gate_bias(client) -> None:
    s = submit(
        client,
        {
            "device": {'kind': "mos_cap", 'parameters': {}},
            "sweep": {"kind": 'cv', 'contact': "gate", "voltages": [-1.0, 1.0]},
        },
    )
    drain(client, s)

    xx = client.get(f"/api/jobs/{s}/fields/1")
    c = json.loads(
        xx.content[4:4 + int.from_bytes(xx.content[:4], 'little')]
    )
    assert c["voltage"] == 1.0
    assert c["index"] == 1

def test_the_fields_of_a_point_that_was_never_solved_are_not_found(
    client,
) -> None:
    info = submit(client, diode_request())
    drain(client, info)
    assert client.get(f"/api/jobs/{info}/fields/9").status_code == 404


def test_the_fields_of_a_running_job_are_refused(client) -> None:
    s = submit(client, diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{s}/fields/0").status_code == 409


def test_the_page_is_served_from_the_root(client) -> None:
    u = client.get("/")
    assert u.status_code == 200
    assert 'text/html' in u.headers['content-type']


def test_the_page_is_revalidated_rather_than_cached(client) -> None:
    num = client.get("/")

    assert num.headers["cache-control"] == "no-cache"

def test_shutting_the_app_down_cancels_what_is_still_solving() -> None:
    ss = JobRegistry()
    with TestClient(create_app(ss)) as z:
        x2 = submit(z, diode_request(voltages=LONG))

    assert ss.status(x2) is JobStatus.CANCELLED

def test_client_scripts_are_served_and_revalidated(client) -> None:
    cc = client.get("/static/js/app.js")

    assert cc.status_code == 200
    assert "javascript" in cc.headers["content-type"]

    assert cc.headers["cache-control"] == "no-cache"

def test_the_page_loads_its_script_rather_than_inlining_it(client) -> None:
    t2 = client.get('/').text

    assert '<script src="/static/js/app.js"></script>' in t2

def test_the_schema_offers_the_stack_regions(client) -> None:
    u = client.get('/api/schema').json()
    assert u['regions']['stack'][0]["dopant"] == 'p'
    assert "pn_diode" not in u["regions"]
def test_the_schema_offers_the_drawing_parts(client) -> None:
    e = client.get('/api/schema').json()['drawings']
    assert set(e) == {"drawing"}
    assert e["drawing"]["blocks"][0]["material"] == "silicon"
    assert e['drawing']["electrodes"][2]['name'] == "gate"


def test_a_stack_the_models_do_not_cover_is_refused_with_its_reason(client) -> None:
    m = [
        {"dopant":"p", 'length':5e-5, 'concentration':1e16},
        {"dopant":"n", "length":5e-5, "concentration":1e21},
    ]
    el = client.post("/api/jobs", json={'device':{'kind':'stack', "parameters":{"regions": m}}, "sweep":{'kind':'iv', "contact":"left", "voltages": [0.1]},},)
    assert el.status_code == 400; assert "region 2" in el.json()['detail']
    assert "references/physics.md" in el.json()['detail']


def test_a_device_can_be_checked_without_solving_it(client) -> None:
    info = client.post("/api/devices/check", json={"kind":'pn_diode'})

    ok = client.post(
        "/api/devices/check",
        json={'kind': "pn_diode", "parameters": {"length": 3e-5}},
    )
    b = client.post(
        "/api/devices/check",
        json={'kind': "pn_diode", "parameters": {'n_nodes': 1001, "h_min": 1e-6}},
    )

    assert info.status_code == 200
    assert ok.status_code == 400 and 'inside' in ok.json()['detail']
    assert b.status_code == 400 and "h_min" in b.json()["detail"]


def test_the_schema_names_each_devices_contacts(client) -> None:
    bb = client.get("/api/schema").json()['contacts']
    assert bb["pn_diode"] == ["anode", 'cathode']
    assert bb["nmos"] == ["source", "drain", "gate", "body"]

    assert set(bb) == set(client.get('/api/schema').json()['devices'])

def test_a_busy_server_answers_503_and_says_to_try_again() -> None:
    with TestClient(create_app(JobRegistry(max_running=0))) as c2:
        item = c2.post("/api/jobs", json=diode_request())

    assert item.status_code == 503; assert 'try again' in item.json()['detail'].lower()


def test_the_page_has_a_favicon_that_is_served(client) -> None:
    u = client.get("/").text
    assert '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">' in u
    r = client.get("/static/favicon.svg")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")


@pytest.mark.parametrize("path, heading", [("/terms", "Terms of use"), ("/privacy", "Privacy")])
def test_the_terms_and_privacy_pages_are_served_and_linked(client, path, heading) -> None:
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert f"<h1>{heading}</h1>" in r.text
    assert '<link rel="icon" href="/static/favicon.svg"' in r.text
    assert "—" not in r.text
    assert f'href="{path}"' in client.get("/").text


def test_every_file_the_page_serves_ships_in_the_installed_package() -> None:
    import fnmatch, pathlib, tomllib
    root = pathlib.Path(__file__).parents[2]
    pats = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]["ddsim"]
    pkg = root / "ddsim"
    miss = [str(f.relative_to(pkg)) for f in (pkg / "api" / "static").rglob("*") if f.is_file() and "__pycache__" not in f.parts and not any(fnmatch.fnmatch(f.relative_to(pkg).as_posix(), x) for x in pats)]
    assert miss == [], f"not in pyproject package-data, so the deployed app 404s on: {miss}"


CONTACTS = ('href="mailto:yashgupta2009yg@gmail.com"', 'href="https://instagram.com/yaa.assshh"', 'href="https://github.com/yaasshh09"')


def test_the_page_has_a_footer_with_the_legal_pages_and_a_contact_dialog(client) -> None:
    u = client.get("/").text
    f = u[u.index("<footer"):u.index("</footer>")]
    for x in ('href="/terms"', 'href="/privacy"', 'id="contact-open"', "© 2026 Yash Gupta"):
        assert x in f, x
    d = u[u.index('<dialog id="contact-card"'):u.index("</dialog>")]
    for x in CONTACTS:
        assert x in d, x


@pytest.mark.parametrize("path", ["/terms", "/privacy"])
def test_the_legal_pages_say_how_to_reach_me(client, path) -> None:
    u = client.get(path).text
    for x in CONTACTS:
        assert x in u, (path, x)



SMALL_CAP = {'n_silicon': 41, 'n_oxide': 3, 'h_min': 2e-7}


def ac_request(voltages, frequencies) -> dict:
    return {"device": {"kind": "mos_cap", "parameters": SMALL_CAP}, "sweep": {"kind": "ac", "contact": "gate", "voltages": voltages, "frequencies": frequencies}}


def test_an_ac_sweep_streams_a_point_per_bias_and_frequency(client) -> None:
    got = drain(client, submit(client, ac_request([-1.0, 1.0], [0.0, 1e6])))
    assert got[-1]["status"] == "done"
    assert [(f["index"], f["frequency"]) for f in got if f["type"] == "ac_point"] == [(0, 0.0), (0, 1e6), (1, 0.0), (1, 1e6)]


def test_an_ac_result_carries_g_and_c_bit_for_bit_with_their_spreads(client) -> None:
    j = submit(client, ac_request([1.0], [0.0, 1e6]))
    drain(client, j)
    body = client.get(f"/api/jobs/{j}/result").json()
    want = ac_sweep(mos_cap(**SMALL_CAP), "gate", [1.0], [0.0, 1e6])
    p, w = body["points"][0], want.points[0]
    assert body["kind"] == "ac"
    assert body["complete"] is True
    assert body["contacts"] == list(w.admittances[0].contacts)
    assert body["frequencies"] == [0.0, 1e6]
    assert body["roundoff"] == ROUNDOFF
    assert p["voltage"] == 1.0
    assert p["G"] == [a.G.tolist() for a in w.admittances]
    assert p["C"] == [None, w.admittances[1].C.tolist()]
    assert p["G_spread"] == [x.tolist() for x in w.G_spread]
    assert p["C_spread"] == [x.tolist() for x in w.C_spread]


def test_the_fields_of_an_ac_point_carry_its_bias(client) -> None:
    j = submit(client, ac_request([-1.0, 1.0], [1e6]))
    drain(client, j)
    xx = client.get(f"/api/jobs/{j}/fields/1")
    assert json.loads(xx.content[4:4 + int.from_bytes(xx.content[:4], 'little')])["voltage"] == 1.0


def test_frequencies_on_a_sweep_without_them_are_refused(client) -> None:
    r = client.post('/api/jobs', json=diode_request(frequencies=[1e3]))
    assert r.status_code == 400
    assert "frequenc" in r.json()["detail"]


STEP_TIMES = [1e-12, 1e-11, 1e-10, 1e-9]


def step_request(voltages=(0.0, 0.3), times=STEP_TIMES, **sweep: Any) -> dict:
    return {"device": {"kind": "pn_diode", "parameters": DIODE}, "sweep": {"kind": "transient", "contact": "anode", "voltages": list(voltages), "times": list(times), **sweep}}


def test_a_transient_streams_a_step_per_accepted_time_step(client) -> None:
    j = submit(client, step_request())
    got = drain(client, j)
    assert got[-1]["status"] == "done"
    steps = [f for f in got if f["type"] == "step"]
    body = client.get(f"/api/jobs/{j}/result").json()
    assert [f["step"] for f in steps] == list(range(1, body["steps"] + 1))
    assert set(STEP_TIMES) <= {f["time"] for f in steps}
    assert steps[-1]["currents"]["anode"] == body["points"][-1]["current"]


def test_a_transient_result_carries_each_sample_at_the_stepped_contact(client) -> None:
    from ddsim.extract.transient import transient_sweep
    j = submit(client, step_request())
    drain(client, j)
    body = client.get(f"/api/jobs/{j}/result").json()
    want = transient_sweep(pn_diode(**DIODE), "anode", [0.0, 0.3], STEP_TIMES)
    assert body["kind"] == "transient"
    assert body["contact"] == "anode"
    assert body["complete"] is True
    assert body["steps"] == want.steps
    assert body["rejected"] == want.rejected
    assert [p["time"] for p in body["points"]] == STEP_TIMES
    assert [p["current"] for p in body["points"]] == [p.currents["anode"] for p in want.points]
    assert [p["voltage"] for p in body["points"]] == [p.voltages["anode"] for p in want.points]


def test_the_fields_of_a_transient_sample_carry_its_time_and_its_bias(client) -> None:
    j = submit(client, step_request(voltages=(0.0, 1.0), settings={"resistance": 10.0, "rtol": 1e-2}))
    drain(client, j)
    body = client.get(f"/api/jobs/{j}/result").json()
    xx = client.get(f"/api/jobs/{j}/fields/2")
    head = json.loads(xx.content[4:4 + int.from_bytes(xx.content[:4], 'little')])
    assert head["time"] == STEP_TIMES[2]
    assert head["voltage"] == body["points"][2]["voltage"]
    assert 0.0 < head["voltage"] < 1.0


def test_times_on_a_sweep_without_them_are_refused(client) -> None:
    r = client.post('/api/jobs', json=diode_request(times=[1e-9]))
    assert r.status_code == 400
    assert "time" in r.json()["detail"]


def test_a_transient_with_one_voltage_is_refused_before_the_job(client) -> None:
    r = client.post('/api/jobs', json=step_request(voltages=(0.3,)))
    assert r.status_code == 400
    assert "two voltages" in r.json()["detail"]


def test_every_point_of_an_iv_result_carries_its_error_estimate(client) -> None:
    from ddsim.extract.error import current_error
    v = submit(client, diode_request(voltages=(0.2, 0.4)))
    frames = drain(client, v)
    w = client.get(f"/api/jobs/{v}/result").json()
    d = pn_diode(**DIODE)
    c = iv_sweep(d, "anode", [0.2, 0.4], models=TransportModels.for_device(d))
    for got, p in zip(w["points"], c.points, strict=True):
        e = current_error(d.with_bias(anode=p.voltage), p.state, "anode")
        assert got["error"] == e.estimate
        assert got["error_linear"] is e.linear
    streamed = [f for f in frames if f["type"] == "error_point"]
    assert [f["index"] for f in streamed] == [0, 1]
    assert [f["estimate"] for f in streamed] == [z["error"] for z in w["points"]]
    assert frames[-1]["type"] == "status"


def test_a_cv_result_carries_the_capacitance_error_estimate(client) -> None:
    from ddsim.extract.error import capacitance_error
    v = submit(client, {"device": {"kind": "mos_cap", "parameters": {}}, "sweep": {"kind": "cv", "contact": "gate", "voltages": [0.0]}})
    drain(client, v)
    w = client.get(f"/api/jobs/{v}/result").json()
    assert w["points"][0]["error"] == capacitance_error(mos_cap().with_bias(gate=0.0), "gate").estimate


def test_ac_and_transient_results_carry_no_error_estimate_yet(client) -> None:
    v = submit(client, diode_request(voltages=(0.2,), kind="ac", frequencies=[1e3]))
    drain(client, v)
    w = client.get(f"/api/jobs/{v}/result").json()
    assert "error" not in w["points"][0]

