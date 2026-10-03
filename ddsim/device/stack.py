from __future__ import annotations
from ddsim.core.config import CONFIG
from  dataclasses import  dataclass
import numpy  as np
from ddsim.device.builder import Device,Material,build_device
from ddsim.device.doping import Layers
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d_at

DOPING_RANGE  = (1e14,
                1e19 )


NODES_INSIDE=CONFIG.mesh.nodes_inside


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
    ok , u = DOPING_RANGE
    for a2,z in enumerate(regions,start =1) :
        if  z.dopant  not in( 'n', 'p'  )  :
            raise ValueError(
                f"region {a2}: the dopant is 'n' or 'p', got "
                f"{z.dopant!r}. For an intrinsic layer, draw a lightly "
                f"doped one instead, say {ok:g} n-type, the bottom of the "
                "range the models are built for."
            )
        if not z.length> 0.0  :
            raise ValueError(
                f"region {a2}: the length must be positive, got "
                f"{z.length:g} cm"
            )
        if not ok<= z.concentration<= u  :
            raise ValueError(
                f"region {a2}: a doping of {z.concentration:g} cm^-3 "
                f"is outside {ok:g} to {u:g} cm^-3, the range the mobility, "
                "recombination and statistics models here are built for (see "
                'references/physics.md).'
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
    n_nodes : int = CONFIG.stack.n_nodes,
    h_min  : float = CONFIG.stack.h_min,
    left_voltage  :  float  =0.0,
    right_voltage  :  float  = 0.0,
    material  :  Material |None = None,
) ->Device :
    _check_regions(regions)
    j = np.cumsum([lst.length for lst in regions])
    h  = [float(f) for f, ( d,  m2  ) in zip(j[  :-   1 ],   zip ( regions[  :-  1 ],  regions[ 1 :], strict   =   True),  strict  =  True) if d.net_doping   !=  m2.net_doping]
    if not h:
        raise ValueError(
            'this stack has no junction: every region has the same doping, so '
            "there is nothing for the mesh to grade towards and no device to "
            "see. Change the doping of one region."
        )

    for m, lst in enumerate(regions, start= 1) :
        if lst.length<(NODES_INSIDE +1)*h_min:
            raise _too_short(m, lst, int(lst.length /h_min))
    res =graded_mesh_1d_at(float(j[-1]), n_nodes, tuple(h), h_min)


    b=np.concatenate([[0.0],j[:-1]])
    for m, (lst, k, f)in enumerate(zip(regions, b, j, strict=True), start= 1)  :
        xx=int(np.count_nonzero((res.x >  k)& (res.x  <  f)))
        if xx <  NODES_INSIDE:
            raise _too_short(m,lst,xx)
    v= (
        OhmicContact(name='left',node=0,voltage=left_voltage),
        OhmicContact(name = "right",node = res.n_nodes - 1,voltage= right_voltage),
    )


    return build_device(
        mesh  =  res,
        doping =  Layers (
            boundaries   =  tuple(float(  out )  for  out  in j [:-  1] ) ,
            values  =  tuple ( z.net_doping for  z  in regions),
        ) ,
        contacts   = v,
        material   =  material,
    )
