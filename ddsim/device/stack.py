"""A 1D stack of doped regions, the device builder of phases/PHASE-7.md Stage 4.

Regions are laid left to right, any number of them, each with a length and a
doping, with an ohmic contact at each end: pn, pin, p+n, npn and whatever a
student invents. Nothing new is solved here. It is the pn_diode recipe with
more than one junction: the same graded mesh generator, the same kind of
doping profile, the same contacts. Drawn as the Phase 2 diode, it is that
diode to the last bit.

Graded at every junction
------------------------
The mesh is graded_mesh_1d_at over the junctions: h_min at every one, one
growth rate for the whole stack, so a thin base between two junctions meets
its neighbours at the same spacing. With one junction that is graded_mesh_1d
itself, which is how the Phase 2 diode drawn as a stack stays that diode.

A boundary between two regions of the same doping is not a junction. Nothing
changes there, so there is nothing to grade towards, and the stack is the
same device as the one with those two regions drawn as one.
"""

from __future__  import  annotations
from  dataclasses import  dataclass
import numpy  as np

from ddsim.device.builder import Device,Material,build_device

from ddsim.device.doping import Layers

from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d_at

DOPING_RANGE  = (1e14,
                1e19 )


"""The doping a region may have [cm^-3], the range docs/01-physics.md states
for the models a 1D device is solved with."""

NODES_INSIDE=2

"""Fewest mesh nodes a region must hold strictly inside it [1]. Two nodes
make three cells, the least that shows a slope inside the region rather
than a straight line from one boundary to the other."""


@dataclass(frozen  = True)




class Region :
    """One doped slab of a stack."""


    dopant : str
    """"n" for donors, "p" for acceptors."""


    length : float


    """Thickness along the stack [cm]."""


    concentration : float

    """Dopant concentration [cm^-3], given positive."""
    @property


    def net_doping(self)->float:
        """Nd - Na [cm^-3], the sign convention of docs/01-physics.md."""
        return self.concentration if self.dopant == "n" else -self.concentration


    def describe (  self  )  ->   str  :
        '''The region as a student drew it, for a refusal to name.'''
        return f"{self.dopant}-type {self.concentration:g} cm^-3, {self.length:g} cm"

PHASE_2_DIODE = (Region("p", 0.5e-4, 1e16), Region("n", 0.5e-4, 1e16))
"""pn_diode's defaults drawn as a stack."""



def _check_regions(regions : tuple[Region, ...]) -> None :
    """Refuse a region that is not a slab of doped silicon the models cover."""

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
    """Doped regions laid left to right, an ohmic contact at each end.

    Args:
        regions: the regions in order from the left contact, at least two,
            with the doping changing at one boundary at least.
        n_nodes: how many mesh points the whole stack is cut into [1].
            Range 51 to 1001. They get shared out between the junctions, so a
            stack with lots of junctions wants more.
        h_min: the smallest mesh spacing, at every junction [cm].
            Range 1e-8 to 1e-6, log. It should be under half the Debye
            length: 20 nm at 1e16 but only 0.65 nm at 1e19, so the 1 nm
            default is too coarse for the most heavily doped regions.
        left_voltage: voltage on the left contact [V]. Range -5 to 1, the
            same as the diode's contacts.
        right_voltage: voltage on the right contact [V]. Range -5 to 1, the
            same as the left contact.
        material: defaults to silicon at 300 K.

    n_nodes is shared between the junctions by how far each one's grading has
    to grow. The h_min rule is the half Debye length one in
    docs/02-numerics.md. The contact ranges copy the diode's, and where a
    cold solve stops converging was measured on the Phase 2 diode, near 1.3 V
    forward, and not on other stacks.
    """
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
