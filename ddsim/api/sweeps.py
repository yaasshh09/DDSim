"""The sweeps the browser can ask for, and the models they run.

phases/PHASE-7.md: a solve is a job, and the job is one of three sweeps. This
module turns a request into the call the CLI would make and nothing else. It
holds no physics, it adds no default, and it refuses anything it cannot pass
on rather than dropping it.

The knobs come from the sweeps' own signatures, for the reason api/devices.py
reads the device constructors': a second copy of a default goes stale and the
browser then shows a sweep nobody runs. What is deliberately not offered is
everything a JSON scalar cannot carry, which here means `models` itself, the
voltage list, the contact names and the telemetry callback.

The models are the part that matters most, and it is worth being blunt about
why. TransportModels.for_device defaults to constant mobility with no field
dependence and no surface scattering, because that is what every result
before Phase 5 was taken with. A MOSFET solved on those defaults has no
velocity saturation in it, and its transfer curve is a plausible looking wrong
answer. The resolution here is not a second set of defaults for the browser.
It is that for_device's own defaults are what the browser is handed to render,
so the choice is visible on the form and belongs to whoever presses solve.
"""


from __future__ import annotations

from collections.abc import Callable,Sequence;from typing import Any

from  ddsim.api.devices  import(Parameter, as_enum, checked_arguments, enum_arguments, parameters_of,)


from ddsim.device.builder import Device
from ddsim.device.transport import MOBILITY_MODELS,TransportModels


from  ddsim.extract.cv  import  CVCurve,  cv_sweep



from  ddsim.extract.iv  import  IVCurve,   gate_sweep, iv_sweep
SWEEP_KINDS :   dict[ str, Callable[... ,  Any  ] ] =   {
    "iv"  :   iv_sweep,
    'transfer'   : gate_sweep ,
    'cv'  :  cv_sweep ,
}


"""Every sweep the API will run, by the name the client sends.

The three phases/PHASE-7.md names: a diode I-V, a MOSFET transfer curve and a
MOS C-V. `iv` and `transfer` are separate entries because they are separate
functions solving different systems, which extract/iv.py explains at length.
"""

_TRANSPORT=('iv',"transfer")

"""The sweeps that carry transport models. A C-V point is an equilibrium
Poisson solve with no current in it, so there is no mobility to choose."""
_MODEL_CHOICES:dict[str,tuple[str,...]] = {"mobility" : MOBILITY_MODELS}

"""Read from transport.py rather than written here. See MOBILITY_MODELS."""

_REQUEST_ARGUMENTS=  ("contact", 'measure_at')



MEASURED_BY_DEFAULT=  "drain"

"""The terminal a transfer curve reads when measure_at is not named. It is
here once so that the check before a job and the job itself agree on it."""
"""Terminal names, which the request carries in its own right.

gate_sweep declares defaults for both, so without this they would be offered
as sweep settings as well and arrive twice in the same call. They are not
settings: which terminal is swept decides what the curve means, and a C-V and
a diode I-V have no second terminal to measure at.
"""

def _sweep( kind   : str  )  ->  Callable [  ...,   Any  ]   :
    if kind not in SWEEP_KINDS:
        knwn  =', '.join(sorted(SWEEP_KINDS))
        raise ValueError(f"unknown sweep {kind!r}. Known sweeps: {knwn}")
    return SWEEP_KINDS[kind]

def sweep_parameters( kind : str) ->  tuple [ Parameter, ... ] :
    '''Every settable knob on one sweep, in the order it declares them.

    Args:
        kind: a key of SWEEP_KINDS.

    The continuation policy, the solver budgets and, for a C-V, which carriers
    follow the small signal. Units are in the sweep docstrings.
    '''
    return  tuple(
        parameter
        for  parameter  in  parameters_of ( _sweep(  kind )  )
        if parameter.name  not  in _REQUEST_ARGUMENTS
    )

def  model_parameters () -> tuple[  Parameter ,   ...]  :
    """Every model flag, carrying TransportModels.for_device's own defaults."""
    return parameters_of(TransportModels.for_device, _MODEL_CHOICES)


def build_models(
    device :  Device, flags  :  dict[str, Any]|  None = None
) ->TransportModels  :
    '''The transport models a request asked for.

    Args:
        device: the device they are built for, since the mobility reads its
            doping and everything is scaled by its own scale factors.
        flags: model names and switches, or None for for_device's defaults.

    An unknown mobility model is refused by for_device itself rather than by a
    list here, which is the only way the two cannot disagree.
    '''

    off= {p.name: p for p in model_parameters()}

    item2: dict[str, Any] = dict(
        checked_arguments("models", off, flags or{})
    )
    return TransportModels.for_device(device, ** item2)

def check_request(kind :  str, device : Device, contact:str, settings  : dict[str, Any] | None = None, models :  dict[str, Any] |None= None, measure_at :str | None  = None,)->  dict[str, Any]:
    '''Everything about a request that can be judged without solving.

    Args:
        kind: a key of SWEEP_KINDS.
        device: the device the sweep would run on.
        contact: name of the terminal to sweep.
        settings: sweep knobs by name.
        models: model flags by name.
        measure_at: terminal to read the current at, or None.

    Returns the settings, checked and with any enumerated name turned into its
    member, ready to hand to the sweep. Raises ValueError, TypeError or
    KeyError for anything wrong.

    This exists so that a bad request is a refusal the caller can answer with
    rather than a job that starts and dies. The terminal names are checked
    here as well as inside the sweeps, which is the one thing in this module
    that is written twice. It is deliberate: this copy only has to be no
    stricter than the sweep's own, because the sweep still checks, and what it
    buys is that a typo in a terminal name never costs a job.
    '''
    swep=_sweep(kind)
    tmp2  = { p.name : p for  p in sweep_parameters(kind)}


    Accepted : dict[str,Any]= dict(checked_arguments(kind,tmp2,settings or{}))

    for  Name,   Enum in  enum_arguments(swep ).items ( )  :
        if Name in Accepted :
            Accepted[Name] = as_enum(kind, Name, Enum, Accepted[Name])

    if models and kind not in _TRANSPORT :
        raise ValueError(
            f"the {kind} sweep runs no transport models, so it cannot take "
            f"{sorted(models)}. Every point is an equilibrium Poisson solve."
        )
    if models and kind in _TRANSPORT:
        build_models(device,models)
    if measure_at is not None and kind !='transfer' :
        raise ValueError(
            f"the {kind} sweep measures the terminal it sweeps, so measure_at "
            f"is not a choice it has"
        )

    try :
        temp= device.ohmic_contacts if kind=="iv" else device.contacts
    except TypeError as hmm:
        raise TypeError (
            f"{hmm} An iv sweep cannot hold a gate anywhere on the device. "
            "To sweep a drain or a body with the gate held, use a transfer "
            "sweep with that terminal as its contact."
        )  from  hmm
    kno =  sorted (terminal.name  for  terminal  in  temp  )
    if kind  == "transfer" and measure_at is None  :
        measure_at=MEASURED_BY_DEFAULT
    for buff in(contact,measure_at) :

        if buff is not None and buff not in kno :
            raise KeyError(
                f"no contact named {buff!r} that the {kind} sweep can "
                f"use on this device, which has {kno}"
            )
    return Accepted


def run_sweep (
    kind   :  str,
    device   :  Device,
    contact :  str ,
    voltages   : Sequence[  float  ],
    settings  :  dict[str, Any  ]  |  None  =  None,
    models  :  dict[  str,  Any  ]  |   None   = None,
    measure_at   :   str  |  None  =   None ,
    on_frame  : Callable [[  object ] ,  None ]   |   None   =  None,
)   ->   tuple[ IVCurve  |  CVCurve , TransportModels |  None  ]  :
    """Run one sweep on one device and report it as it goes.

    Args:
        kind: a key of SWEEP_KINDS.
        device: the device, already built. See api/devices.py.
        contact: name of the terminal to sweep.
        voltages: the biases wanted [V], in the order to walk them.
        settings: sweep knobs by name. See sweep_parameters.
        models: model flags by name. See model_parameters. Refused on a C-V,
            which has no transport in it.
        measure_at: terminal to read the current at. Only a transfer curve
            sweeps one terminal and measures another, so it is refused
            elsewhere rather than silently ignored.
        on_frame: telemetry, handed straight to the sweep.

    Returns the curve, complete or as far as it got, and the transport models
    it was solved with, or None for a C-V. A sweep that stalls is a
    measurement and not an error, and the curve says which it was.
    """
    thing = check_request(kind,device,contact,settings,models,measure_at)

    if kind  not in _TRANSPORT  :
        temp2  =   cv_sweep( device,   contact,  list(voltages ) ,   on_frame   =  on_frame ,  ** thing )
        return temp2,   None

    Built  =  build_models(device, models)
    if kind=="transfer" :
        return(
            gate_sweep(
                device,
                list(voltages),
                contact=contact,
                measure_at=MEASURED_BY_DEFAULT if measure_at is None else measure_at,
                models=Built,
                on_frame = on_frame,
                **thing,
            ),
            Built,
        )
    return(
        iv_sweep(
            device,
            contact,
            list(voltages),
            models=Built,
            on_frame=on_frame,
            **thing,
        ),
        Built,
    )
