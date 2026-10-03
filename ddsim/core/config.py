from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Newton:
    max_iterations: int = 50
    residual_atol : float = 1e-12
    residual_rtol: float = 1e-10
    update_tol: float =1e-10
    stagnation_window : int = 4


@dataclass(frozen=True)
class Gummel:
    max_iterations: int = 200
    update_tol : float = 1e-08


@dataclass(frozen=True)
class Bias:
    max_iterations : int = 30
    max_psi_step: float = 5.0
    max_surface_sweeps: int = 20
    surface_rtol : float = 1e-08
    ramp_step: float = 0.25
    gummel_cycles: int = 3
    retry_cycles : int = 5


@dataclass(frozen=True)
class Continuation:
    growth: float = 1.5
    max_attempts: int = 200


@dataclass(frozen=True)
class Sweeps:
    iv_step : float = 0.05
    gate_step: float = 0.1
    rolloff_step: float = 0.05
    drain_low : float = 0.05
    drain_high: float = 1.0
    slope_decades: float = 2.0
    reference_current : float = 1e-07


@dataclass(frozen=True)
class Mesh:
    max_ratio: float = 1.5
    ratio_tolerance : float = 1e-14
    degenerate_tolerance: float = 1e-12
    score_slack: float = 1e-09
    full_cell_tol : float = 1e-09
    nodes_inside: int = 2
    node_budget : int = 20000


@dataclass(frozen=True)
class PnDiode:
    Na: float = 1e+16
    Nd: float = 1e+16
    length : float = 0.0001
    junction: float = 5e-05
    n_nodes: int = 201
    h_min : float = 1e-07


@dataclass(frozen=True)
class Stack:
    n_nodes: int = 201
    h_min : float = 1e-07


@dataclass(frozen=True)
class MosCap:
    substrate_doping: float = -1e+16
    t_ox : float = 1e-06
    t_si: float = 0.0002
    width: float = 1e-05
    nx : int = 3
    n_silicon: int = 121
    n_oxide: int = 5
    h_min : float = 5e-08


@dataclass(frozen=True)
class Nmos:
    L_gate: float = 0.0001
    sd_length : float = 4e-05
    contact_length: float = 2e-05
    substrate_doping: float = -1e+17
    sd_peak : float = 1e+20
    x_j: float = 1.5e-05
    lateral_diffusion: float = 1e-05
    t_ox : float = 2e-06
    t_si: float = 0.0001
    n_contact: int = 6
    n_sd : int = 12
    n_channel: int = 16
    n_silicon: int = 101
    n_oxide : int = 33
    h_min_x: float = 2e-07
    h_min_y: float = 6.25e-09


@dataclass(frozen=True)
class Drawing:
    nx: int = 63
    ny : int = 133
    h_min_x: float = 2e-07
    h_min_y : float = 6.25e-09
    degenerate_doping_top: float = 1e+20


@dataclass(frozen=True)
class Numerics:
    exp_limit: float = 700.0
    quadrature_order : int = 96
    quadrature_tail: float = 60.0
    joyce_dixon_max_u: float = 8.0
    inversion_steps : int = 6
    bernoulli_series_cutoff: float = 0.0001
    bernoulli_series_cutoff_db : float = 0.1
    bernoulli_asymptote_cutoff_db: float = 80.0
    equilibrium_max_psi_step : float = 5.0
    flux_floor_margin: float = 16.0
    density_reference : float = 1.0


@dataclass(frozen=True)
class Server:
    port: int = 8000
    queue_size : int = 4096
    shutdown_timeout: float = 30.0
    quiet_poll : float = 0.25
    public_max_running: int = 2
    public_keep_for : float = 1800.0
    public_time_limit: float = 300.0


@dataclass(frozen=True)
class Config:
    newton: Newton = field(default_factory=Newton)
    gummel : Gummel = field(default_factory=Gummel)
    bias: Bias = field(default_factory=Bias)
    continuation : Continuation = field(default_factory=Continuation)
    sweeps: Sweeps = field(default_factory=Sweeps)
    mesh : Mesh = field(default_factory=Mesh)
    pn_diode: PnDiode = field(default_factory=PnDiode)
    stack : Stack = field(default_factory=Stack)
    mos_cap: MosCap = field(default_factory=MosCap)
    nmos : Nmos = field(default_factory=Nmos)
    drawing: Drawing = field(default_factory=Drawing)
    numerics : Numerics = field(default_factory=Numerics)
    server: Server = field(default_factory=Server)


CONFIG = Config()
