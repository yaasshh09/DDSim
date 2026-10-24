from __future__ import annotations
from functools import cache
import numpy as np
from ddsim.device.bjt import BASE, COLLECTOR, EMITTER, npn_2d
from ddsim.device.transport import TransportModels
from ddsim.extract.bjt import GummelPlot, gummel_plot, ideality
from ddsim.extract.iv import terminal_currents
from ddsim.mesh.mesh2d import Mesh2D

V_BE = [0.4, 0.5, 0.6, 0.7, 0.8]


@cache
def plot() -> GummelPlot:
    d = npn_2d()
    g = gummel_plot(d, V_BE, models=TransportModels.for_device(d), f_t=False)
    assert g.complete, g.message
    return g


def test_the_contacts_sit_where_they_should() -> None:
    d = npn_2d()
    assert isinstance(d.mesh, Mesh2D)
    top = d.mesh.y_axis.x[-1]
    by = {c.name: c for c in d.contacts}
    for name in (EMITTER, BASE):
        assert np.allclose(d.mesh.node_y[list(by[name].nodes)], top)
    assert np.allclose(d.mesh.node_y[list(by[COLLECTOR].nodes)], 0.0)
    assert d.mesh.node_x[list(by[EMITTER].nodes)].max() < d.mesh.node_x[list(by[BASE].nodes)].min()
    assert np.all(d.net_doping.data[list(by[EMITTER].nodes)] > 0.0)
    assert np.all(d.net_doping.data[list(by[BASE].nodes)] < 0.0)
    assert np.all(d.net_doping.data[list(by[COLLECTOR].nodes)] > 0.0)


def test_the_emitter_sits_inside_the_base_inside_the_collector() -> None:
    d = npn_2d()
    m = d.mesh
    assert isinstance(m, Mesh2D)
    column = d.net_doping.data.reshape(m.ny, m.nx)[:, 0]
    signs = np.sign(column[::-1])
    changes = np.flatnonzero(np.diff(signs) != 0)
    assert changes.size == 2, "under the emitter, going down, the doping should read n+, then p, then n"


def test_the_gummel_plot_runs_and_the_currents_balance() -> None:
    g = plot()
    assert np.all(g.collector > 0.0) and np.all(g.base > 0.0)
    assert 5.0 < float(np.max(g.beta)) < 1e4, "with Boltzmann statistics and no bandgap narrowing a 1e20 emitter barely takes any holes, so beta is in the thousands, not the hundreds"
    d = npn_2d()
    p = g.points[-1]
    m = TransportModels.for_device(d)
    i = terminal_currents(d.with_bias(emitter=-p.v_be), p.state, m)
    assert abs(i[EMITTER] + i[BASE] + i[COLLECTOR]) <= 1e-8 * abs(i[COLLECTOR])


def test_the_2d_collector_current_stays_close_to_ideal() -> None:
    g = plot()
    n = ideality(g.v_be, g.collector, (0.4, 0.6))
    assert 1.0 <= n < 1.1, f"ideality {n:.3f}: the graded base makes it a little above one as the emitter junction edge moves, but not much"
