from __future__ import annotations
import argparse, contextlib, datetime, io
import math, os
import sys
from typing import Any
from devsim import(
    add_2d_contact,
    add_2d_interface,
    add_2d_mesh_line,
    add_2d_region,
    contact_equation,
    create_2d_mesh,
    create_device,
    delete_device,
    delete_mesh,
    edge_average_model,
    edge_from_node_model,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_interface_list,
    get_node_model_values,
    get_parameter,
    node_model,
    node_solution,
    set_node_values,
    set_parameter,
    solve,
    vector_gradient,
)
from devsim.python_packages.model_create import(CreateContactNodeModel, CreateEdgeModel, CreateEdgeModelDerivatives, CreateNodeModel, CreateNodeModelDerivative, CreateSolution,)
from  devsim.python_packages.simple_physics  import(
    CreateOxideContact ,
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion ,
    CreateSiliconDriftDiffusionAtContact ,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact ,
    GetContactBiasName ,
    GetContactNodeModelName,
    celec_model ,
    chole_model,
    ece_name,
    hce_name,
)
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from  tests.regression.devsim_gen import parameters as  P

BULK ="bulk"

OXIDE= "oxide"

SOURCE = "source"

DRAIN='drain'


GATE = "gate"

BODY='body'



AIR= 1e-6
GOLDEN  = os.path.join("data", 'golden')
set_parameter ( name  = 'threads_available',  value  =   1)

SOLUTIONS  :  dict[str, tuple[str, ...]]  = {BULK : ("Potential", 'Electrons', "Holes"), OXIDE :  ('Potential', ),}


def devsim_version()->str:
    import devsim

    return str(getattr(devsim,'__version__','unknown'))



@contextlib.contextmanager



def quiet()  :


    with contextlib.redirect_stdout(io.StringIO()) :
        yield

def note(  message   : str ) ->  None   :
    print(f"    {message}", file = sys.stderr, flush = True)


def build_mesh(
    benchmark  : P.MosfetBenchmark,
    device :str,
    refine:float=1.0,
    refine_y :float |None =  None,
    process :dict[str, float]|None= None,
)  -> None :

    xs  =  device
    process=P.MOSFET_PROCESS if process is None else process
    m  =  process[ 'sd_length'  ]
    g =process["contact_length"]
    b2 =process['t_si']
    s  = process['t_ox']
    r =  process["x_j"]
    thing =benchmark.L_gate
    d2  =  2.0* m+thing

    t= refine if refine_y is None else refine_y
    f  =  benchmark.devsim_h_contact /  refine
    cnt  =  benchmark.devsim_h_junction  /   refine
    i=benchmark.devsim_h_channel/refine;res2 =benchmark.devsim_h_surface / t
    v = benchmark.devsim_h_depth  /  t
    j =s/(benchmark.devsim_oxide_cells*t)

    create_2d_mesh(mesh=xs)

    for m2,e,num in((0.0,f,f), (g,f,f), (m,cnt,cnt), (m+0.5* thing,i,i), (m +thing,cnt,cnt), (d2 -g,f,f), (d2,f,f),):
        add_2d_mesh_line(mesh= xs,dir= "x",pos= m2,ns= e,ps =num)

    for m2,e,num in(
        (- AIR,AIR,AIR),
        (0.0,AIR,0.2 * b2),
        (b2 -r,v,v),
        (b2,res2,j),
        (b2+s,j,AIR),
        (b2 + s +AIR,AIR,AIR),
    ):
        add_2d_mesh_line(  mesh =  xs,
                   dir  =  "y" ,
                   pos  =   m2,
               ns   =  e,
           ps = num)


    add_2d_region(mesh=xs,region=BULK,material="Silicon",yl=0.0,yh =b2)


    add_2d_region(
        mesh=  xs, region  =OXIDE, material  ="Oxide", yl =b2, yh= b2+s
    )
    add_2d_region(mesh  =xs, region= 'air_bot', material ='metal', yl =- AIR, yh = 0.0)

    add_2d_region(
        mesh =xs,
        region =  "air_top",
        material  = "metal",
        yl= b2  +  s,
        yh= b2 +s + AIR,
    )
    add_2d_contact(mesh = xs, name = BODY, material ='metal', region  = BULK, yl  = 0.0, yh =0.0)
    add_2d_contact(
        mesh= xs,
        name= SOURCE,
        material="metal",
        region = BULK,
        xl=0.0,
        xh=g,
        yl= b2,
        yh=b2,
    )
    add_2d_contact(mesh =xs, name= DRAIN, material="metal", region= BULK, xl=d2 -g, xh=d2, yl=b2, yh= b2,)
    add_2d_contact(
        mesh  = xs,
        name=GATE,
        material = "metal",
        region=  OXIDE,
        xl =m,
        xh = m +  thing,
        yl=b2+ s,
        yh  = b2+s,
    )
    add_2d_interface(mesh  = xs, name =  "si_ox", region0 = BULK, region1 = OXIDE, yl =b2, yh=  b2,)
    finalize_mesh(mesh=xs)
    create_device(mesh= xs, device  = device)
    check_mesh_landed(device, b2)


def check_mesh_landed( device : str, t_si  : float  )  ->  None   :

    h= get_node_model_values(device=device,region=BULK,name ="y")
    w =  max(h)
    if abs(w  - t_si)  > 1e-14 * t_si  :
        raise  RuntimeError (
            f"{device}: the top silicon row is at {w:.17e} cm and the "
            f"interface is at {t_si:.17e}, a gap of {t_si - w:.3e} cm. "
            "devsim merged the interface mesh line into its neighbour, so the "
            "surface contacts and the si_ox interface have no nodes. Choose a "
            'surface spacing the rows below it can grade onto.'
        )
    c  =  set (get_contact_list (device  = device  ) )
    thing = { BODY,  SOURCE,   DRAIN ,  GATE }  -  c
    if  thing   :
        raise RuntimeError(
            f"{device}: contacts {sorted(thing)} were asked for and not "
            f"created. devsim has {sorted(c)}."
        )

    dd = set(get_interface_list(device =  device))
    if "si_ox" not in dd:
        raise RuntimeError(
            f"{device}: the si_ox interface was not created. devsim has "
            f"{sorted(dd)}."
        )



def node_count(device  : str) -> int:

    return len(get_node_model_values(device= device, region= BULK, name =  "x"))



def set_material_parameters(device:str)->None:
    v2   =   {
        "Permittivity" : P.EPS_R_SI  *  P.EPS_0 ,
        "ElectronCharge" :   P.Q ,
        "n_i"   : P.N_I,
        'T'  :   P.T ,
        "kT"   :   P.K_B   *  P.T ,
        'V_t'  : P.V_T ,
        'mu_n' :  P.MU_N,
        "mu_p" : P.MU_P,
        'n1'  :  P.N_I ,
        "p1"   :  P.N_I ,
    }
    for zz,y in v2.items() :
        set_parameter(device =device,region=BULK,name = zz,value =y)
    for zz, y in(("Permittivity", P.EPS_R_OX*P.EPS_0), ('ElectronCharge', P.Q),) :
        set_parameter(device=device, region = OXIDE, name =  zz, value= y)



def set_doping(benchmark  : P.MosfetBenchmark, device :str, process  : dict[str, float]  | None =None,)-> None  :
    process= P.MOSFET_PROCESS if process is None else process
    w,thing= P.implant_shape(process)
    cc  =  2.0 * process[  'sd_length'] +  benchmark.L_gate
    def  implant(x :  str )  ->  str  :
        return(
            f"{process['sd_peak']:.16e} * 0.5 * "
            f"erfc(({x} - {process['sd_length']:.16e})/{thing:.16e}) * "
            f"exp(-pow({process['t_si']:.16e} - y, 2)/(2*pow({w:.16e}, 2)))"
        )

    node_model (
        device  =   device ,
        region  =  BULK,
        name   = 'NetDoping',
        equation  = (
            f"{process['substrate_doping']:.16e} + "
            + implant ( 'x' )
            + " + "
            + implant( f"({cc:.16e} - x)" )
        ),
    )


def set_lifetimes( device  :  str  )  ->  None :
    for a, v, jj in(("taun", P.TAU_N_MAX, P.TAU_N_MIN), ('taup', P.TAU_P_MAX, P.TAU_P_MIN),) :
        CreateNodeModel(
            device,
            BULK,
            a,
            f"{jj:.16e} + ({v:.16e} - {jj:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})",
        )



def joyce_dixon(u:  str)  ->  str:
    y2, r, j, vv  =  P.JOYCE_DIXON
    tt=f"(min({u}, {P.JOYCE_DIXON_MAX_U:.16e}))"
    return(
        f"({y2:.16e}*{tt} + {r:.16e}*pow({tt},2) + "
        f"{j:.16e}*pow({tt},3) + {vv:.16e}*pow({tt},4))"
    )
def set_full_stack_parameters(device : str) ->None :
    d2: dict[str, float] =  {
        "Nc": P.NC_300,
        'Nv'  :P.NV_300,
        'v_sat_n': P.V_SAT_N,
        'v_sat_p': P.V_SAT_P,
        "beta_n":  P.BETA_N,
        'beta_p'  :  P.BETA_P,
        "E_perp_floor" : P.E_PERP_FLOOR,
    }
    for d, tmp in enumerate(('mu_min', "mu_d", "N_ref", 'arora_A'))  :
        d2[f"{tmp}_n"] =P.ARORA_N[d]
        d2[f"{tmp}_p"]=  P.ARORA_P[d]
    for g, a in P.LOMBARDI_N.items() :
        d2[f"lom_{g}_n"]= a
    for g,a in P.LOMBARDI_P.items():

        d2[f"lom_{g}_p"] = a
    for tmp, a in d2.items()  :
        set_parameter(  device  =  device, region  =   BULK,   name  = tmp,   value  =  a )


def create_bulk_mobility(device:  str)  ->None :
    for d in("n","p"):
        CreateNodeModel(
            device,
            BULK,
            f"mu_arora_{d}",
            f"mu_min_{d} + mu_d_{d}/(1 + "
            f"pow(abs(NetDoping)/N_ref_{d}, arora_A_{d}))",
        )


def create_surface_mobility(device  :   str)  ->   None  :

    node_solution(device  =  device, region =  BULK,  name   = "E_perp"  )
    CreateNodeModel(device, BULK, "E_perp_used", "max(E_perp, E_perp_floor)")
    CreateNodeModel(
        device,  BULK ,  'N_surface' ,  f"max(abs(NetDoping), {P.N_I:.16e})"
    )

    for dd in('n',"p"):

        vv = (
            f"lom_B_{dd}/E_perp_used + "
            f"lom_C_{dd}*pow(N_surface, lom_tau_{dd})*"
            f"pow(E_perp_used, -1.0/3.0)/pow(T/300, lom_kappa_{dd})"
        )

        t= (
            f"lom_A_{dd} + lom_alpha_{dd}*(Electrons + Holes)*"
            f"pow(N_surface, -lom_eta_{dd})"
        )
        tmp2   = f"lom_delta_{dd}*pow(E_perp_used, -({t}))"
        xx = f"mu_arora_{dd}"
        CreateNodeModel(device,BULK,f"mu_ac_{dd}",vv)
        CreateNodeModel(device,BULK,f"mu_sr_{dd}",tmp2)
        CreateNodeModel(
            device,
            BULK,
            f"mu_low_{dd}_model",
            f"{xx}*mu_ac_{dd}*mu_sr_{dd} / "
            f"({xx}*mu_ac_{dd} + {xx}*mu_sr_{dd} + "
            f"mu_ac_{dd}*mu_sr_{dd})",
        )
        node_solution(device  = device, region  =BULK, name= f"mu_low_{dd}")
    refresh_surface_mobility(device  )

SURFACE_RTOL=1e-8

SURFACE_SWEEPS=20


def  refresh_surface_mobility(  device  :  str)  ->  float   :
    vector_gradient (device = device,   region =  BULK ,  node_model =  "Potential" , calc_type =  'default')
    i= get_node_model_values(device=device,region=BULK,name= 'Potential_grady')
    set_node_values(
        device= device,
        region = BULK,
        name= "E_perp",
        values = [abs(rr)for rr in i],
    )
    c  =  0.0
    for idx in('n',"p") :
        row=f"mu_low_{idx}"

        tmp3= get_node_model_values(device =device,region=BULK,name=row)

        set_node_values(
            device  = device, region =  BULK, name=  row, init_from =f"{row}_model"
        )
        tmp =get_node_model_values(device =  device, region = BULK, name  =row)
        for val2,k in zip(tmp3,tmp,strict=True):
            if k !=0.0:
                c =  max(c, abs(k - val2) / abs(k))
    return  c
def create_edge_mobility(device  :  str ) ->   None :

    for b in("n", "p"):
        edge_average_model(
            device = device,
            region = BULK,
            node_model =  f"mu_low_{b}",
            edge_model =  f"mu_lf_{b}",
            average_type  = 'arithmetic',
        )

        j=(
            f"pow(mu_lf_{b}*ElectricField/v_sat_{b}, 2) + 1e-300"
        )

        d =(
            f"mu_lf_{b}*pow(1 + pow({j}, 0.5*beta_{b}), "
            f"-1.0/beta_{b})"
        )

        tt= f"mu_ct_{b}"


        CreateEdgeModel(device,BULK,tt,d)
        CreateEdgeModelDerivatives(device, BULK, tt, d, "Potential")



def create_degenerate_currents(device:str)->None:

    for item, z2, dd, v2 in(
        ('n', 'Electrons', "Nc", '-'),
        ('p', 'Holes', "Nv", "+"),
    )  :
        r =  joyce_dixon (f"{z2}/{dd}"  ); w=f"Potential {v2} V_t*{r}"
        CreateNodeModel(device,BULK,f"Potential_{item}",w)
        for t2 in("Potential", z2)  :

            CreateNodeModelDerivative(
                device,   BULK,   f"Potential_{item}",  w,  t2
            )

        edge_from_node_model(
            device  = device, region = BULK, node_model =  f"Potential_{item}"
        )
        for t2 in("Potential",z2):
            edge_from_node_model(
                device  =   device,
                region =   BULK ,
                node_model   =  f"Potential_{item}:{t2}",
            )
        h= f"(Potential_{item}@n0 - Potential_{item}@n1)/V_t"
        CreateEdgeModel(device , BULK ,  f"vdiff_{item}",   h  )
        for t2 in('Potential',z2):
            for yy,v2 in(("@n0",""),("@n1","-")):
                CreateEdgeModel(
                    device,
                    BULK,
                    f"vdiff_{item}:{t2}{yy}",
                    f"{v2}Potential_{item}:{t2}{yy}/V_t",
                )
        CreateEdgeModel(device, BULK, f"Bern01_{item}", f"B(vdiff_{item})")
        for t2 in("Potential",z2):
            for yy in('@n0', "@n1"):
                CreateEdgeModel (
                    device,
                    BULK,
                    f"Bern01_{item}:{t2}{yy}",
                    f"dBdx(vdiff_{item}) * vdiff_{item}:{t2}{yy}",
                )
    d=("ElectronCharge*mu_ct_n*EdgeInverseLength*V_t*kahan3(" "Electrons@n1*Bern01_n, Electrons@n1*vdiff_n, -Electrons@n0*Bern01_n)")
    t =(
        "-ElectronCharge*mu_ct_p*EdgeInverseLength*V_t*kahan3("
        'Holes@n1*Bern01_p, -Holes@n0*Bern01_p, -Holes@n0*vdiff_p)'
    )
    for b, cc in(
        ("ElectronCurrent", d),
        ("HoleCurrent", t),
    ):
        CreateEdgeModel(device, BULK, b, cc)
        for t2  in(  "Electrons",   "Holes", 'Potential')  :
            CreateEdgeModelDerivatives(device, BULK, b, cc, t2)

def degenerate_contact_expressions()->tuple[str,str,str]:

    i  = f"exp(-{joyce_dixon(f'{celec_model}/Nc')})"
    j =  f"exp(-{joyce_dixon(f'{chole_model}/Nv')})"
    m =f"(n_i^2*{i}*{j})"
    xs = f"ifelse(NetDoping > 0, {celec_model}, {m}/{chole_model})"
    k=f"ifelse(NetDoping < 0, {chole_model}, {m}/{celec_model})"
    e=(
        f"ifelse(NetDoping > 0, "
        f"-V_t*(log({celec_model}/n_i) + {joyce_dixon(f'{celec_model}/Nc')}), "
        f"+V_t*(log({chole_model}/n_i) + {joyce_dixon(f'{chole_model}/Nv')}))"
    )
    return xs,k,e



def create_degenerate_contact(device :str,contact:str) ->None :

    cc, v, el = degenerate_contact_expressions()
    r2  =  f"Potential -{GetContactBiasName(contact)} + {el}"
    CreateContactNodeModel(
        device, contact, GetContactNodeModelName(contact), r2
    )
    CreateContactNodeModel(
        device, contact, f"{GetContactNodeModelName(contact)}:Potential", '1'
    )



    for out,r,j in(
        (f"{contact}nodeelectrons",f"Electrons - ({cc})",'Electrons'),
        (f"{contact}nodeholes",f"Holes - ({v})",'Holes'),
    ):
        CreateContactNodeModel(device,
                     contact,
                    out,
          r)
        CreateContactNodeModel(device, contact, f"{out}:{j}", '1')
    for s2,cur,m in(
        (ece_name,f"{contact}nodeelectrons","ElectronCurrent"),
        (hce_name,f"{contact}nodeholes",'HoleCurrent'),
    ):
        contact_equation(
            device  =device,
            contact  = contact,
            name = s2,
            node_model= cur,
            edge_current_model  =m,
        )

def gate_potential(v_gate:float,work_function:float=P.PHI_M_N_POLY) ->float:
    return v_gate + (  P.PHI_M_MIDGAP  -  work_function  )

def seed_potential(device  : str)  -> None  :
    node_model(
        device =  device,
        region = BULK,
        name= "PotentialNeutral",
        equation  = (
            f"{P.V_T:.16e} * sgn(NetDoping) * log((abs(NetDoping) + "
            f"pow(NetDoping*NetDoping + 4*{P.N_I:.16e}*{P.N_I:.16e}, 0.5)) "
            f"/ (2*{P.N_I:.16e}))"
        ),
    )
    set_node_values(device= device, region  = BULK, name ="Potential", init_from =  'PotentialNeutral')




def  build_physics(
    device :   str ,
    v_gate   :   float,
    v_drain : float,
    models  :   str =  P.REDUCED_MODELS,
    work_function   : float  =  P.PHI_M_N_POLY ,
)  ->  None :
    for  k  in(  BULK ,   OXIDE  )  :
        CreateSolution (device,  k,  "Potential")
    CreateSiliconPotentialOnly(device, BULK)
    seed_potential(device)
    CreateOxidePotentialOnly(device, OXIDE, "log_damp")

    for x in(BODY,SOURCE,DRAIN) :
        set_parameter(  device  =  device, name   =  f"{x}_bias",   value   = 0.0 )
        CreateSiliconPotentialOnlyContact(device,
                  BULK,
               x)
    set_parameter(
        device=device,
        name=f"{GATE}_bias",
        value=gate_potential(v_gate,work_function),
    )
    CreateOxideContact(device,OXIDE,GATE)
    CreateSiliconOxideInterface(device,"si_ox")
    settle( device,  poisson_only  =  True  )

    for row, e in(
        ('Electrons', 'IntrinsicElectrons'),
        ("Holes", 'IntrinsicHoles'),
    ):


        CreateSolution(device, BULK, row)
        set_node_values(device= device, region =  BULK, name = row, init_from =  e)
    set_lifetimes(device)


    if  models  ==  P.REDUCED_MODELS :
        CreateSiliconDriftDiffusion(device, BULK, mu_n ="mu_n", mu_p = 'mu_p')
        for x in(BODY,SOURCE,DRAIN):
            CreateSiliconDriftDiffusionAtContact(device,BULK,x)

    else :
        set_full_stack_parameters(device)
        create_bulk_mobility(device)
        create_surface_mobility(  device  )
        create_edge_mobility(device )
        CreateSiliconDriftDiffusion(device,BULK,mu_n ='mu_ct_n',mu_p ="mu_ct_p")
        create_degenerate_currents(  device)
        for x in(BODY,
                SOURCE,
                   DRAIN) :
            create_degenerate_contact(device, x)
        global  _surface_is_live
        _surface_is_live  =  True

    settle(  device  )

    ramp_to(device,DRAIN,v_drain)
def snapshot(device:str,poisson_only: bool=False)->dict[tuple[str,str],list]:

    s = ({BULK  :  ("Potential", ), OXIDE : ('Potential', )}  if poisson_only else SOLUTIONS)
    return{( d ,  m )  :   list(get_node_model_values( device =   device ,  region  =  d,   name  = m )) for  d,   y in  s.items() for  m  in y}

def restore(device: str,saved:dict[tuple[str,str],list])->None:

    for(num, y), lst in saved.items()  :
        set_node_values(device  =device, region = num, name=y, values = lst)
def potential_move(before : dict[tuple[str, str], list], after  : dict[tuple[str, str], list])-> float:

    stuff =0.0


    for d, bb in before.items() :
        if d[1] != "Potential" :
            continue
        for xs, val2 in  zip (  bb,   after [ d ] , strict =  True )  :
            stuff=max(stuff,abs(xs- val2))
    return stuff



def sane(device:str) -> bool:

    psi =get_node_model_values(device = device,region=BULK,name ='Potential')
    if max(abs(idx)for idx in psi)>5.0 :


        return False
    for tmp3 in("Electrons", 'Holes'):
        s=get_node_model_values(device=device,region =BULK,name=tmp3)
        if min(s)<  0.0 or max(s)> 1e22 :
            return False
    return  True

_surface_is_live=False

SOLVE_TOLERANCES  :   tuple[  float,  ...  ]  =  (  1e-8,   1e-6 , 1e-4)

EQUILIBRIUM_ITERATIONS= 1000




def solve_once(rescue: int | None=None) -> None:


    for j,ss in enumerate(SOLVE_TOLERANCES):

        try :
            solve(
                type =  'dc',
                absolute_error  =  1e30 ,
                relative_error = ss ,
                maximum_iterations =  100,
                maximum_error =   1e40,
            )
            return
        except Exception  :
            if  j   +   1  == len(  SOLVE_TOLERANCES  ) :

                if rescue is None :
                    raise
                solve(type='dc',absolute_error=1e30, relative_error =ss, maximum_iterations=rescue ,maximum_error= 1e40)  # devsim crawls here, needs the 1000 or it dies


def settle(device:  str, poisson_only :  bool = False, passes: int = 400, tol:float  =  1e-9, balance_tol  : float |None= None, stall : int=25,) ->  int :

    w = snapshot(device,poisson_only)
    s=  _surface_is_live and not poisson_only
    r  =  0
    h  =  float("inf")

    g =float("inf")
    b  =  0
    ret  =   0.0
    d =False
    m  = float(  "inf" )
    tmp2 =  float ( 'inf'  )
    u = 0

    for cur in range(passes) :
        if s :
            r+=1
            thing = refresh_surface_mobility(device)
            if(thing< SURFACE_RTOL or r>= SURFACE_SWEEPS):

                s= False
                g   =  float('inf'  )


                b  =  0 ; ret  =  0.0
                if thing >= SURFACE_RTOL:
                    note(
                        f"surface mobility still moving at "
                        f"{thing:.3e} after {r} "
                        "refreshes, frozen there"
                    )
        solve_once(EQUILIBRIUM_ITERATIONS if poisson_only else None)
        k  = snapshot(device, poisson_only)
        h=potential_move(w,
               k)
        w = k


        if not d and h >=tol :

            if h< g:
                g, b,   ret  =  h , 0 ,   0.0
                continue
            b +=1
            ret  =  max(ret, h)


            if b<stall:
                continue

            if ret>=SETTLE_FLOOR  :


                raise RuntimeError(
                    f"stalled at {g:.3e} V for {stall} passes, worst "
                    f"{ret:.3e} V, last move {h:.3e} V"
                )
            note(
                f"settled on the solver floor, {ret:.3e} V over "
                f"{stall} passes"
            )


        d =True
        if not poisson_only and not sane(device):
            raise RuntimeError('settled on a state no bias can produce')
        if poisson_only or balance_tol is None:
            return cur+ 1


        m  =terminal_imbalance(device)
        if m < balance_tol  :
            return cur +1
        if m< BALANCE_PROGRESS*tmp2:
            tmp2, u  =  m, 0
            continue

        tmp2 =min(tmp2,m)
        u+=1
        if u>= stall :
            return cur + 1

    if d:
        return passes
    raise  RuntimeError(
        f"did not settle in {passes} passes, last move {h:.3e} V, "
        f"imbalance {m:.3e}"
    )

GROWTH_STREAK = 4


def ramp_to(device:str, contact : str, target :float, step:float= 0.1, min_step: float =1e-4, max_step:float= 0.1,)->None :
    c = get_parameter(device = device, name=f"{contact}_bias"); ys =0
    while abs(target- c)> 1e-12:
        b=snapshot(device)
        r=min(step,abs(target- c))
        m  =   c   +  math.copysign(r,   target -  c  )
        set_parameter(device =  device, name = f"{contact}_bias", value  = m)
        try:
            with quiet()  :
                settle(device)
        except Exception:
            restore(device,b)
            set_parameter(device=device,name=f"{contact}_bias",value = c)
            step *= 0.5
            ys  = 0
            if step  <   min_step   :
                raise  RuntimeError(
                    f"{contact} stuck at {c:g} V heading to {target:g} V "
                    f"on {device}: no step above {min_step:g} V converges"
                )   from  None
            continue
        c=m


        ys  +=  1
        if  ys  >=   GROWTH_STREAK   :
            step=min(2.0 * step,
                     max_step)
            ys= 0



def terminal_current(device:str,contact : str)->float:
    return get_contact_current(
        device = device, contact= contact, equation =  ece_name
    )  +get_contact_current(device=device, contact =contact, equation=  hce_name)

SETTLE_FLOOR = 1e-5

BALANCE_PROGRESS  =  0.9

BALANCE_TOL =   1e-3



def terminal_imbalance(device: str) ->float :
    flag =  terminal_current(device, DRAIN)
    dd=terminal_current(device,SOURCE)
    v=max(abs(flag),abs(dd))
    return 0.0  if v  ==  0.0 else abs(flag   +  dd  )  / v




def device_name(
    benchmark : P.MosfetBenchmark, drain  :float, refine :float =1.0
) ->  str  :
    return f"{benchmark.name}_d{drain:g}_r{refine:g}".replace(".","_")



def  transfer_curve(benchmark  : P.MosfetBenchmark, drain   :  float, refine :  float  = 1.0, refine_y  : float  |  None   =   None,)   ->  tuple[  list[  dict[ str ,   Any]  ] ,   int ] :

    global _surface_is_live
    _surface_is_live=False
    y = device_name(benchmark,drain,refine)
    with quiet() :
        build_mesh(benchmark,y,refine=refine,refine_y=refine_y)
    set_material_parameters(y)
    set_doping(benchmark, y)
    with quiet() :
        build_physics(y,benchmark.gate_voltages[0],drain,benchmark.models)
    m : list[dict[str, Any]]=[]


    for s in benchmark.gate_voltages:
        ramp_to (y , GATE, gate_potential ( s))
        with quiet()  :
            settle(y, balance_tol  =BALANCE_TOL)

        m.append ({"gate"  :  s, "drain" : terminal_current( y, DRAIN), "source" :  terminal_current ( y, SOURCE), 'body' :  terminal_current(y , BODY  ),})


        print(
            f"    Vg={s:+.3f} V  Id={m[-1]['drain']:+.6e} A/cm",
            flush = True,
        )
    idx  =  node_count ( y)

    _surface_is_live   =  False
    delete_device(device =y)
    delete_mesh( mesh   =   y  )
    return m, idx



def sweep(
    benchmark:P.MosfetBenchmark,
    refine :float=1.0,
    refine_y : float |None = None,
)->tuple[list[dict[str,Any]],list[dict[str,Any]],int]:
    print(f"  {benchmark.name}: Vd = {benchmark.drain_low} V" ,   flush   =  True  )
    tt,aa=transfer_curve(benchmark,benchmark.drain_low,refine =refine,refine_y= refine_y)
    print(f"  {benchmark.name}: Vd = {benchmark.drain_high} V",flush=True)


    tmp2, _=transfer_curve(benchmark, benchmark.drain_high, refine  = refine, refine_y = refine_y)
    return tt,tmp2,aa


MESH_NOISE_FACTOR  = 3.0
def  relative_difference(coarse  :  list[dict [  str, Any] ],   fine : list[dict[ str,  Any ]  ])   -> tuple[ float ,  float , int ]   :


    kk =  0.0
    m  = 0.0
    jj =0
    for  val2, rr  in zip(coarse,   fine, strict  =   True  ) :
        if(abs(val2["drain"]) < P.CURRENT_FLOOR_MOSFET and abs(rr['drain'])<P.CURRENT_FLOOR_MOSFET) :
            jj  +=  1
            continue
        z = max(abs(val2['drain']), abs(rr["drain"]))
        u = abs(val2['drain'] - rr['drain'])/  z
        t   =  MESH_NOISE_FACTOR   * max(row_imbalance( val2 ), row_imbalance (rr  ) )
        if u<=t:
            jj+= 1
            continue
        if u >kk :

            kk,m=u,val2["gate"]
    return kk,m,jj


def row_imbalance(row:dict[str,Any])-> float:
    m =  max(abs(row['drain']), abs(row['source']))
    return  0.0 if  m  ==  0.0 else  abs(row['drain'  ] + row[ 'source'])  /  m


def write_csv(benchmark:P.MosfetBenchmark, low :  list[dict[str, Any]], high : list[dict[str, Any]], nodes  :int, mesh_check :tuple[float, float, int] |None, path :  str,) -> None  :

    info = P.MOSFET_PROCESS
    h, r = P.implant_shape(info)
    res=datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')
    d2:list[str] =[
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_mosfet.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {res}",
        f"# tolerance: {benchmark.tolerance}",
        f"# L_gate: {benchmark.L_gate:.6e}",
        f"# drain low: {benchmark.drain_low:.6e}",
        f"# drain high: {benchmark.drain_high:.6e}",
    ]

    d2.extend(f"# {s}: {ii:.6e}" for s, ii in info.items())
    d2.extend(
        [
            f"# implant sigma: {h:.6e}",
            f"# implant edge: {r:.6e}",
            f"# devsim silicon nodes: {nodes}",
            f"# devsim h_junction: {benchmark.devsim_h_junction:.6e}",
            f"# devsim h_channel: {benchmark.devsim_h_channel:.6e}",
            f"# devsim h_contact: {benchmark.devsim_h_contact:.6e}",
            f"# devsim h_surface: {benchmark.devsim_h_surface:.6e}",
            f"# devsim h_depth: {benchmark.devsim_h_depth:.6e}",
            f"# devsim oxide cells: {benchmark.devsim_oxide_cells}",
        ]
    )
    if  mesh_check  is not None :
        tmp2,t,a = mesh_check
        d2.append(
            f"# mesh convergence: {tmp2:.3e} worst relative change in drain "
            f"current when every spacing is halved, at {t:+g} V of gate, "
            f"with {a} of {len(low)} points skipped as unmeasurable"
        )

    d2.append('# notes: '+benchmark.notes)
    d2.append('# models:')
    d2.extend ("#   "  + y  for y  in  P.mosfet_model_summary( benchmark.models  ))
    d2.append(
        "# columns: gate bias [V], then drain and source current [A/cm] at "
        'the low drain bias and then at the high one'
    )
    d2.append('gate_voltage,drain_low,source_low,drain_high,source_high')
    for u, z in zip(low, high, strict  = True)  :
        d2.append(
            f"{u['gate']:.10g},{u['drain']:.12e},{u['source']:.12e},"
            f"{z['drain']:.12e},{z['source']:.12e}"
        )

    with open(path,'w',encoding= "utf-8",newline="\n") as j:
        j.write("\n".join(d2  )   + "\n"  )

def main(  )  ->   int  :

    m =  argparse.ArgumentParser(description=  "Generate MOSFET golden data")
    m.add_argument("names", nargs = "*", default  =None, help  = 'benchmark names to generate. Default is all of them.',)
    m.add_argument(
        "--no-mesh-check",
        action="store_true",
        help='skip the halved mesh run, which doubles the runtime.',
    )


    m.add_argument ("--out", default   =  GOLDEN, help  =  'directory to write into. Default is data/golden.',)
    i =m.parse_args()


    c  =  P.MOSFET_BENCHMARKS
    if i.names  :

        prev  =  sorted(set(i.names) -  set(P.MOSFET_BY_NAME))
        if prev  :
            print( f"no such benchmark: {prev}" ,  file = sys.stderr )
            print(f"known: {sorted(P.MOSFET_BY_NAME)}", file = sys.stderr)
            return  1
        c = tuple(P.MOSFET_BY_NAME[m2]for m2 in i.names)
    os.makedirs (  i.out,   exist_ok =  True )
    for a2 in c :

        print(
            f"{a2.name}: L_gate = {a2.L_gate * 1e7:g} nm, "
            f"{len(a2.gate_voltages)} gate biases on two curves",
            flush  = True,
        )
        x ,  k, u =  sweep( a2 )

        xs =os.path.join(i.out,f"{a2.name}.csv")
        write_csv(  a2,  x, k , u , None,  xs  )


        s=None
        if not i.no_mesh_check:

            print(f"{a2.name}: repeating on a halved mesh", flush=True)
            zz, val, _= sweep(a2, refine =2.0)
            s  = max(
                relative_difference(x, zz),
                relative_difference(k, val),
                key= lambda check: check[0],
            )
            print (
                f"{a2.name}: worst relative change {s[0]:.3e} "
                f"at {s[1]:+g} V, {s[2]} points skipped",
                flush  =   True ,
            )


            write_csv(a2,x,k,u,s,xs)
        print(f"{a2.name}: wrote {xs}",flush= True)
    return 0
if __name__ =='__main__' :
    raise  SystemExit (  main(  ))
