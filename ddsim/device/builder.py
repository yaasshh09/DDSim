from __future__  import  annotations



from  dataclasses import dataclass ,  replace
from functools import cached_property
import numpy as np, numpy.typing  as  npt

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors

from ddsim.device.doping import Coordinates, DopingProfile
from ddsim.device.regions import RegionMap

from ddsim.discretize.boundary import Contact,GateContact,SemiconductorContact
from ddsim.discretize.geometry import ScaledMesh

from ddsim.mesh.mesh1d  import  Mesh1D
from ddsim.mesh.mesh2d import Mesh2D


from  ddsim.physics.statistics  import Degeneracy

AnyMesh  =  Mesh1D   |  Mesh2D


@dataclass(frozen =True)




class Material :

    name :str

    T : float

    eps :  float
    n_i  : float


    @classmethod
    def silicon(cls, T : float  =  C.T_ROOM) ->Material:

        return  cls( name = 'silicon' ,  T =  T , eps  = C.eps_Si(  ),  n_i  =  C.n_i(T ))


@dataclass(frozen =True)

class Device:


    mesh :  AnyMesh

    doping : DopingProfile
    material : Material

    contacts  :  tuple [Contact , ...]
    scale :  ScaleFactors
    regions: RegionMap | None =None

    degenerate  : bool  = False

    @cached_property
    def  net_doping( self )   ->   Field   :
        Values= self.doping(self.node_coordinates)
        if self.regions is not None:
            Values  =  np.where(  self.regions.semiconductor_volume >   0.0 ,  Values,   0.0)
        return Field(Values , "cm^-3", ScalingState.PHYSICAL , Location.NODE, name   =  "net_doping" ,)

    @property
    def node_coordinates(self) ->Coordinates:
        dep=None if isinstance(self.mesh,Mesh1D)else self.mesh.node_y
        return Coordinates(self.node_x, dep)

    @property
    def node_x(  self )   ->  npt.NDArray [ np.float64  ] :

        if  isinstance(  self.mesh ,   Mesh1D ) :
            return self.mesh.x
        return self.mesh.node_x

    @property
    def dimension(self)->int :
        return  1 if isinstance ( self.mesh,  Mesh1D)   else 2
    @cached_property
    def scaled_mesh(self)->  ScaledMesh  :
        if isinstance(self.mesh,Mesh1D):
            return self.mesh.scaled(self.scale)
        if self.regions is None:
            return self.mesh.scaled(self.scale)
        return self.mesh.scaled(
            self.scale,
            eps_r =self.regions.eps_r,
            semiconductor_face= self.regions.semiconductor_face,
        )
    @cached_property
    def charge_volume_scaled(self) ->  npt.NDArray[np.float64] :

        if self.regions is None :
            return self.scaled_mesh.volume
        return np.asarray(
            self.regions.semiconductor_volume /self.scale.x_0** self.dimension
        )


    @cached_property
    def semiconductor_contacts(self) ->  tuple[SemiconductorContact, ...] :
        return tuple(
            contact
            for contact in self.contacts
            if not isinstance(contact, GateContact)
        )

    @cached_property
    def ohmic_contacts(self)->tuple[SemiconductorContact,
               ...]:
        for buf in  self.contacts   :
            if isinstance(buf,GateContact):
                raise TypeError(
                    f"contact {buf.name!r} is a {type(buf).__name__}, "
                    "and this path handles contacts that touch semiconductor "
                    "only. The coupled transport solve pins psi, n and p at "
                    'every node of a contact, and a gate sits on an insulator '
                    'where there is no doping to read and no carrier to pin.'
                )
        return self.semiconductor_contacts  # same thing, gates already bailed above

    @cached_property
    def carrier_free_nodes(self)  -> tuple[int, ...] :

        if self.regions is None :

            return()

        return tuple(int(node)for node in self.regions.oxide_nodes)

    @property
    def mesh_1d(self)  ->  Mesh1D :
        if  not isinstance(self.mesh,   Mesh1D  )  :


            raise TypeError(
                "this path is 1D and the device carries a "
                f"{type(self.mesh).__name__}. The coupled Newton solve and "
                "the current extraction work in both; this is the Gummel "
                "path, which slices edges contiguously."
            )
        return self.mesh
    @cached_property
    def net_doping_scaled(self)  -> Field :
        return self.net_doping.to_scaled(self.scale)
    @cached_property
    def degeneracy(self) -> Degeneracy|None:

        if not self.degenerate:

            return None


        return Degeneracy.for_silicon(self.scale.C_0, self.material.T)


    def with_bias(self,**voltages :float) ->Device:
        konwn  = {con.name for con in self.contacts}

        Unknown= sorted(set(voltages)-konwn)
        if Unknown :
            raise KeyError(
                f"no contact named {Unknown} on this device, which has "
                f"{sorted(konwn)}"
            )



        conacts= tuple(replace(con,voltage=voltages.get(con.name,con.voltage)) for con in self.contacts)
        return replace(self, contacts = conacts)


    def __repr__(self)->str:
        k2 = ', '.join(
            f"{contact.name}={contact.voltage:g}V" for contact in self.contacts
        )
        if isinstance(self.mesh, Mesh1D)  :

            vals=f"length={self.mesh.length:.3e} cm"
        else  :
            vals =(
                f"size={self.mesh.x_axis.length:.3e} by "
                f"{self.mesh.y_axis.length:.3e} cm"
            )
        return(
            f"Device {self.material.name} {self.mesh.n_nodes} nodes "
            f"{vals} contacts=({k2})"
        )

def build_device(mesh :AnyMesh, doping: DopingProfile, contacts:tuple[Contact,...], material : Material | None=None, C_0 : float|None =None, regions:RegionMap|None = None, degenerate: bool =False,)->Device:
    if material is None :
        material = Material.silicon()

    if not contacts  :
        raise ValueError("a device needs at least one contact")

    for w in contacts:
        for nod in w.nodes:
            if not 0 <=nod< mesh.n_nodes:
                raise  IndexError(
                    f"contact {w.name!r} sits on node {nod}, "
                    f"but the mesh has {mesh.n_nodes} nodes"
                )

    if regions is not None:

        if regions.semiconductor_volume.size!=mesh.n_nodes:
            raise  ValueError(
                f"the region map covers {regions.semiconductor_volume.size} "
                f"nodes but the mesh has {mesh.n_nodes}. A region map belongs "
                'to the mesh it was built on.'
            )
        if np.asarray(regions.eps_r).size  != mesh.n_edges:
            raise ValueError(
                f"the region map carries {np.asarray(regions.eps_r).size} edge "
                f"permittivities but the mesh has {mesh.n_edges} edges. A "
                "region map belongs to the mesh it was built on."
            )



    zip = [w.name for w in contacts]
    if len(  set ( zip  ) )  !=   len(  zip ) :
        raise ValueError(f"contact names must be unique, got {zip}")

    sca=ScaleFactors.for_silicon(T= material.T, C_0=  material.n_i if C_0 is None else C_0, eps= material.eps,)


    return  Device(
        mesh =  mesh,
        doping =   doping ,
        material  = material,
        contacts   = contacts,
        scale   =   sca ,
        regions  =  regions,
        degenerate  =  degenerate,
    )
