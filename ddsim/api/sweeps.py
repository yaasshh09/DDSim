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


_TRANSPORT=('iv',"transfer")

_MODEL_CHOICES:dict[str,tuple[str,...]] = {"mobility" : MOBILITY_MODELS}

_REQUEST_ARGUMENTS=  ("contact", 'measure_at')



MEASURED_BY_DEFAULT=  "drain"

def _sweep( kind   : str  )  ->  Callable [  ...,   Any  ]   :
    if kind not in SWEEP_KINDS:
        knwn  =', '.join(sorted(SWEEP_KINDS))
        raise ValueError(f"unknown sweep {kind!r}. Known sweeps: {knwn}")
    return SWEEP_KINDS[kind]

def sweep_parameters( kind : str) ->  tuple [ Parameter, ... ] :
    return  tuple(
        parameter
        for  parameter  in  parameters_of ( _sweep(  kind )  )
        if parameter.name  not  in _REQUEST_ARGUMENTS
    )

def  model_parameters () -> tuple[  Parameter ,   ...]  :
    return parameters_of(TransportModels.for_device, _MODEL_CHOICES)


def build_models(
    device :  Device, flags  :  dict[str, Any]|  None = None
) ->TransportModels  :

    off= {p.name: p for p in model_parameters()}

    item2: dict[str, Any] = dict(
        checked_arguments("models", off, flags or{})
    )
    return TransportModels.for_device(device, ** item2)

def check_request(kind :  str, device : Device, contact:str, settings  : dict[str, Any] | None = None, models :  dict[str, Any] |None= None, measure_at :str | None  = None,)->  dict[str, Any]:
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
