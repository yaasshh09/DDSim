from __future__ import annotations
import numpy as np, pytest
from scipy.sparse import coo_matrix
from ddsim.device.builder import build_device
from ddsim.device.doping import Uniform
from ddsim.device.transport import TransportModels, initial_state
from ddsim.discretize.boundary import OhmicContact
from ddsim.discretize.coupled import (
    UNKNOWNS_PER_NODE,
    Unknown,
    coupled_mass,
    coupled_residual,
    edge_drop,
    pack,
    unpack,
)
from ddsim.mesh.mesh1d import graded_mesh_1d
from ddsim.physics.mobility import diffusivity_at
from tests.reference.complexstep import complex_step_jacobian

N_NODES = 40


@pytest.fixture
def device():
    m = graded_mesh_1d(1e-4, N_NODES, refine_at=0.3e-4, h_min=2e-7)
    return build_device(
        mesh=m,
        doping=Uniform(1e16),
        contacts=(
            OhmicContact(name="left", node=0, voltage=0.0),
            OhmicContact(name='right', node=N_NODES - 1, voltage=0.0),
        ),
    )


@pytest.fixture
def models(device):
    return TransportModels.for_device(device)


@pytest.fixture
def sizes(device):
    s = device.scale
    return device.mesh.h / s.x_0, device.charge_volume_scaled


@pytest.fixture
def bumped(device):
    g = initial_state(device)
    t = np.linspace(0.0, 1.0, N_NODES)
    n = g.n.data * (1.0 + 0.3 * np.exp(-((t - 0.4) / 0.1) ** 2))
    p = g.p.data * (1.0 + 2e11 * np.exp(-((t - 0.6) / 0.15) ** 2))
    return pack(g.psi.data, n, p)


def dense_mass(volume):
    r, c, v = coupled_mass(volume)
    k = UNKNOWNS_PER_NODE * volume.size
    return coo_matrix((v, (r, c)), shape=(k, k)).toarray()


def residual(sizes, device, models):
    h, vol = sizes

    def F(x):
        return coupled_residual(h=h, volume=vol, x=x, net_doping=device.net_doping_scaled.data, Dn=models.Dn, Dp=models.Dp, recombination=models.recombination,)

    return F


def test_mass_is_what_a_backward_euler_step_adds(device, models, sizes, bumped):
    _, vol = sizes
    F = residual(sizes, device, models)
    old = bumped.copy()
    dt = 3.7e-2

    def stepped(x):
        _, n, p = unpack(x)
        _, n0, p0 = unpack(old)
        out = F(x)
        out[Unknown.N::UNKNOWNS_PER_NODE] += vol * (n - n0) / dt
        out[Unknown.P::UNKNOWNS_PER_NODE] += vol * (p - p0) / dt
        return out

    a = complex_step_jacobian(stepped, bumped)
    b = complex_step_jacobian(F, bumped)
    got = dense_mass(vol)

    np.testing.assert_allclose(a, b + got / dt, rtol=1e-13, atol=1e-15 * float(np.max(np.abs(a))))
    assert np.all(got[Unknown.PSI::UNKNOWNS_PER_NODE, :] == 0.0)


def test_carriers_diffuse_and_recombine_forward_in_time(device, models, sizes, bumped):
    h, vol = sizes
    F = residual(sizes, device, models)
    psi, n, p = unpack(bumped)
    assert np.all(psi == psi[0]), "the bar is not flat, so B(0) = 1 does not hold"

    m = dense_mass(vol)
    rows = np.flatnonzero(np.diag(m))
    rate = np.zeros_like(bumped)
    rate[rows] = -F(bumped)[rows] / np.diag(m)[rows]
    _, dn, dp = unpack(rate)

    R = np.asarray(models.recombination.rate(n, p), dtype=np.float64)
    Dn = np.broadcast_to(diffusivity_at(models.Dn, edge_drop(psi), h), h.shape)
    Dp = np.broadcast_to(diffusivity_at(models.Dp, edge_drop(psi), h), h.shape)

    def inflow(D, c):
        k = np.zeros(N_NODES)
        for i in range(N_NODES - 1):
            q = D[i] * (c[i + 1] - c[i]) / h[i]
            k[i] += q
            k[i + 1] -= q
        return k

    want_n = inflow(Dn, n) / vol - R
    want_p = inflow(Dp, p) / vol - R
    np.testing.assert_allclose(dn, want_n, rtol=1e-10, atol=1e-10 * float(np.max(np.abs(want_n))))
    np.testing.assert_allclose(dp, want_p, rtol=1e-10, atol=1e-10 * float(np.max(np.abs(want_p))))
    assert dn[np.argmax(n)] < 0.0 and dp[np.argmax(p)] < 0.0
