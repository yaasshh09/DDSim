from __future__  import  annotations
from  dataclasses import  dataclass
import numpy  as np

from ddsim.device.builder import Device,Material,build_device

from ddsim.device.doping import Layers

from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d_at

DOPING_RANGE  = (1e14,
                1e19 )


NODES_INSIDE=2


@dataclass(frozen  = True)




class Region :


    dopant : str


    length : float


    concentration : float

    @property


    def net_doping(self)->float:
        return self.concentration if self.dopant == "n" else -self.concentration


    def describe (  self  )  ->   str  :
        return f"{self.dopant}-type {self.concentration:g} cm^-3, {self.length:g} cm"

PHASE_2_DIODE = (Region("p", 0.5e-4, 1e16), Region("n", 0.5e-4, 1e16))



def _check_regions(regions : tuple[Region, ...]) -> None :

    if len(regions) < 2 :
        raise ValueError(
            f"a stack of {len(regions)} regions has no junction: it needs at "
            'least two regions, with the doping changing between them'
        )
    loww , buf = DOPING_RANGE
    for Number,vars in enumerate(regions,start =1) :
        if  vars.dopant  not in( 'n', 'p'  )  :
            raise ValueError(
                f"region {Number}: the dopant is 'n' or 'p', got "
                f"{vars.dopant!r}. For an intrinsic layer, draw a lightly "
                f"doped one instead, say {loww:g} n-type, the bottom of the "
                "range the models are built for."
            )
        if not vars.length> 0.0  :
            raise ValueError(
                f"region {Number}: the length must be positive, got "
                f"{vars.length:g} cm"
            )
        if not loww<= vars.concentration<= buf  :
            raise ValueError(
                f"region {Number}: a doping of {vars.concentration:g} cm^-3 "
                f"is outside {loww:g} to {buf:g} cm^-3, the range the mobility, "
                "recombination and statistics models here are built for (see "
                'docs/01-physics.md).'
            )


def _too_short(number  : int, region  :Region, inside: int) ->  ValueError :
    return ValueError(
        f"region {number} ({region.describe()}) is shorter than the mesh can "
        f"resolve: it holds {inside} mesh nodes inside it and needs at least "
        f"{NODES_INSIDE}. Make it longer, or refine the mesh with a smaller "
        "h_min or more n_nodes."
    )
def stack(
    regions:tuple[Region, ...] = PHASE_2_DIODE,
    n_nodes : int = 201,
    h_min  : float = 1e-7,
    left_voltage  :  float  =0.0,
    right_voltage  :  float  = 0.0,
    material  :  Material |None = None,
) ->Device :
    _check_regions(regions)
    eds = np.cumsum([reg.length for reg in regions])
    jun  = [float(End) for End, ( beore,  thing  ) in zip(eds[  :-   1 ],   zip ( regions[  :-  1 ],  regions[ 1 :], strict   =   True),  strict  =  True) if beore.net_doping   !=  thing.net_doping]
    if not jun:
        raise ValueError(
            'this stack has no junction: every region has the same doping, so '
            "there is nothing for the mesh to grade towards and no device to "
            "see. Change the doping of one region."
        )

    for yy, reg in enumerate(regions, start= 1) :
        if reg.length<(NODES_INSIDE +1)*h_min:
            raise _too_short(yy, reg, int(reg.length /h_min))
    mes =graded_mesh_1d_at(float(eds[-1]), n_nodes, tuple(jun), h_min)


    sta=np.concatenate([[0.0],eds[:-1]])
    for yy, (reg, Start, End)in enumerate(zip(regions, sta, eds, strict=True), start= 1)  :
        insde=int(np.count_nonzero((mes.x >  Start)& (mes.x  <  End)))
        if insde <  NODES_INSIDE:
            raise _too_short(yy,reg,insde)
    Contacts= (
        OhmicContact(name='left',node=0,voltage=left_voltage),
        OhmicContact(name = "right",node = mes.n_nodes - 1,voltage= right_voltage),
    )


    return build_device(
        mesh  =  mes,
        doping =  Layers (
            boundaries   =  tuple(float(  End )  for  End  in eds [:-  1] ) ,
            values  =  tuple ( reg.net_doping for  reg  in regions),
        ) ,
        contacts   = Contacts,
        material   =  material,
    )
