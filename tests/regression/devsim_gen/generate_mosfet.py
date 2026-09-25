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

    mes  =  device
    process=P.MOSFET_PROCESS if process is None else process
    sd_lngth  =  process[ 'sd_length'  ]
    temp =process["contact_length"]
    tsi =process['t_si']
    to  = process['t_ox']
    XJ =  process["x_j"]
    LL =benchmark.L_gate
    Width  =  2.0* sd_lngth+LL

    myvar= refine if refine_y is None else refine_y
    h  =  benchmark.devsim_h_contact /  refine
    hj  =  benchmark.devsim_h_junction  /   refine
    dat=benchmark.devsim_h_channel/refine;h_surfce =benchmark.devsim_h_surface / myvar
    hdepth = benchmark.devsim_h_depth  /  myvar
    h_oxde =to/(benchmark.devsim_oxide_cells*myvar)

    create_2d_mesh(mesh=mes)

    for poss,Ns,w in((0.0,h,h), (temp,h,h), (sd_lngth,hj,hj), (sd_lngth+0.5* LL,dat,dat), (sd_lngth +LL,hj,hj), (Width -temp,h,h), (Width,h,h),):
        add_2d_mesh_line(mesh= mes,dir= "x",pos= poss,ns= Ns,ps =w)

    for poss,Ns,w in(
        (- AIR,AIR,AIR),
        (0.0,AIR,0.2 * tsi),
        (tsi -XJ,hdepth,hdepth),
        (tsi,h_surfce,h_oxde),
        (tsi+to,h_oxde,AIR),
        (tsi + to +AIR,AIR,AIR),
    ):
        add_2d_mesh_line(  mesh =  mes,
                   dir  =  "y" ,
                   pos  =   poss,
               ns   =  Ns,
           ps = w)


    add_2d_region(mesh=mes,region=BULK,material="Silicon",yl=0.0,yh =tsi)


    add_2d_region(
        mesh=  mes, region  =OXIDE, material  ="Oxide", yl =tsi, yh= tsi+to
    )
    add_2d_region(mesh  =mes, region= 'air_bot', material ='metal', yl =- AIR, yh = 0.0)

    add_2d_region(
        mesh =mes,
        region =  "air_top",
        material  = "metal",
        yl= tsi  +  to,
        yh= tsi +to + AIR,
    )
    add_2d_contact(mesh = mes, name = BODY, material ='metal', region  = BULK, yl  = 0.0, yh =0.0)
    add_2d_contact(
        mesh= mes,
        name= SOURCE,
        material="metal",
        region = BULK,
        xl=0.0,
        xh=temp,
        yl= tsi,
        yh=tsi,
    )
    add_2d_contact(mesh =mes, name= DRAIN, material="metal", region= BULK, xl=Width -temp, xh=Width, yl=tsi, yh= tsi,)
    add_2d_contact(
        mesh  = mes,
        name=GATE,
        material = "metal",
        region=  OXIDE,
        xl =sd_lngth,
        xh = sd_lngth +  LL,
        yl=tsi+ to,
        yh  = tsi+to,
    )
    add_2d_interface(mesh  = mes, name =  "si_ox", region0 = BULK, region1 = OXIDE, yl =tsi, yh=  tsi,)
    finalize_mesh(mesh=mes)
    create_device(mesh= mes, device  = device)
    check_mesh_landed(device, tsi)


def check_mesh_landed( device : str, t_si  : float  )  ->  None   :

    Rows= get_node_model_values(device=device,region=BULK,name ="y")
    Top =  max(Rows)
    if abs(Top  - t_si)  > 1e-14 * t_si  :
        raise  RuntimeError (
            f"{device}: the top silicon row is at {Top:.17e} cm and the "
            f"interface is at {t_si:.17e}, a gap of {t_si - Top:.3e} cm. "
            "devsim merged the interface mesh line into its neighbour, so the "
            "surface contacts and the si_ox interface have no nodes. Choose a "
            'surface spacing the rows below it can grade onto.'
        )
    con  =  set (get_contact_list (device  = device  ) )
    misssing = { BODY,  SOURCE,   DRAIN ,  GATE }  -  con
    if  misssing   :
        raise RuntimeError(
            f"{device}: contacts {sorted(misssing)} were asked for and not "
            f"created. devsim has {sorted(con)}."
        )

    inteerfaces = set(get_interface_list(device =  device))
    if "si_ox" not in inteerfaces:
        raise RuntimeError(
            f"{device}: the si_ox interface was not created. devsim has "
            f"{sorted(inteerfaces)}."
        )



def node_count(device  : str) -> int:

    return len(get_node_model_values(device= device, region= BULK, name =  "x"))



def set_material_parameters(device:str)->None:
    slicon   =   {
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
    for Name,t2 in slicon.items() :
        set_parameter(device =device,region=BULK,name = Name,value =t2)
    for Name, t2 in(("Permittivity", P.EPS_R_OX*P.EPS_0), ('ElectronCharge', P.Q),) :
        set_parameter(device=device, region = OXIDE, name =  Name, value= t2)



def set_doping(benchmark  : P.MosfetBenchmark, device :str, process  : dict[str, float]  | None =None,)-> None  :
    process= P.MOSFET_PROCESS if process is None else process
    Sigma,out2= P.implant_shape(process)
    range  =  2.0 * process[  'sd_length'] +  benchmark.L_gate
    def  implant(x :  str )  ->  str  :
        return(
            f"{process['sd_peak']:.16e} * 0.5 * "
            f"erfc(({x} - {process['sd_length']:.16e})/{out2:.16e}) * "
            f"exp(-pow({process['t_si']:.16e} - y, 2)/(2*pow({Sigma:.16e}, 2)))"
        )

    node_model (
        device  =   device ,
        region  =  BULK,
        name   = 'NetDoping',
        equation  = (
            f"{process['substrate_doping']:.16e} + "
            + implant ( 'x' )
            + " + "
            + implant( f"({range:.16e} - x)" )
        ),
    )


def set_lifetimes( device  :  str  )  ->  None :
    for nam, tauu_max, bb in(("taun", P.TAU_N_MAX, P.TAU_N_MIN), ('taup', P.TAU_P_MAX, P.TAU_P_MIN),) :
        CreateNodeModel(
            device,
            BULK,
            nam,
            f"{bb:.16e} + ({tauu_max:.16e} - {bb:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})",
        )



def joyce_dixon(u:  str)  ->  str:
    a11, a22, a33, acc  =  P.JOYCE_DIXON
    arr=f"(min({u}, {P.JOYCE_DIXON_MAX_U:.16e}))"
    return(
        f"({a11:.16e}*{arr} + {a22:.16e}*pow({arr},2) + "
        f"{a33:.16e}*pow({arr},3) + {acc:.16e}*pow({arr},4))"
    )
def set_full_stack_parameters(device : str) ->None :
    val: dict[str, float] =  {
        "Nc": P.NC_300,
        'Nv'  :P.NV_300,
        'v_sat_n': P.V_SAT_N,
        'v_sat_p': P.V_SAT_P,
        "beta_n":  P.BETA_N,
        'beta_p'  :  P.BETA_P,
        "E_perp_floor" : P.E_PERP_FLOOR,
    }
    for inex, nam in enumerate(('mu_min', "mu_d", "N_ref", 'arora_A'))  :
        val[f"{nam}_n"] =P.ARORA_N[inex]
        val[f"{nam}_p"]=  P.ARORA_P[inex]
    for Key, Value in P.LOMBARDI_N.items() :
        val[f"lom_{Key}_n"]= Value
    for Key,Value in P.LOMBARDI_P.items():

        val[f"lom_{Key}_p"] = Value
    for nam, Value in val.items()  :
        set_parameter(  device  =  device, region  =   BULK,   name  = nam,   value  =  Value )


def create_bulk_mobility(device:  str)  ->None :
    for carirer in("n","p"):
        CreateNodeModel(
            device,
            BULK,
            f"mu_arora_{carirer}",
            f"mu_min_{carirer} + mu_d_{carirer}/(1 + "
            f"pow(abs(NetDoping)/N_ref_{carirer}, arora_A_{carirer}))",
        )


def create_surface_mobility(device  :   str)  ->   None  :

    node_solution(device  =  device, region =  BULK,  name   = "E_perp"  )
    CreateNodeModel(device, BULK, "E_perp_used", "max(E_perp, E_perp_floor)")
    CreateNodeModel(
        device,  BULK ,  'N_surface' ,  f"max(abs(NetDoping), {P.N_I:.16e})"
    )

    for car in('n',"p"):

        aco = (
            f"lom_B_{car}/E_perp_used + "
            f"lom_C_{car}*pow(N_surface, lom_tau_{car})*"
            f"pow(E_perp_used, -1.0/3.0)/pow(T/300, lom_kappa_{car})"
        )

        bar= (
            f"lom_A_{car} + lom_alpha_{car}*(Electrons + Holes)*"
            f"pow(N_surface, -lom_eta_{car})"
        )
        rou   = f"lom_delta_{car}*pow(E_perp_used, -({bar}))"
        Bulk = f"mu_arora_{car}"
        CreateNodeModel(device,BULK,f"mu_ac_{car}",aco)
        CreateNodeModel(device,BULK,f"mu_sr_{car}",rou)
        CreateNodeModel(
            device,
            BULK,
            f"mu_low_{car}_model",
            f"{Bulk}*mu_ac_{car}*mu_sr_{car} / "
            f"({Bulk}*mu_ac_{car} + {Bulk}*mu_sr_{car} + "
            f"mu_ac_{car}*mu_sr_{car})",
        )
        node_solution(device  = device, region  =BULK, name= f"mu_low_{car}")
    refresh_surface_mobility(device  )

SURFACE_RTOL=1e-8

SURFACE_SWEEPS=20


def  refresh_surface_mobility(  device  :  str)  ->  float   :
    vector_gradient (device = device,   region =  BULK ,  node_model =  "Potential" , calc_type =  'default')
    gra= get_node_model_values(device=device,region=BULK,name= 'Potential_grady')
    set_node_values(
        device= device,
        region = BULK,
        name= "E_perp",
        values = [abs(chr)for chr in gra],
    )
    thing  =  0.0
    for carreir in('n',"p") :
        nam=f"mu_low_{carreir}"

        Before= get_node_model_values(device =device,region=BULK,name=nam)

        set_node_values(
            device  = device, region =  BULK, name=  nam, init_from =f"{nam}_model"
        )
        buff =get_node_model_values(device =  device, region = BULK, name  =nam)
        for dat,sorted in zip(Before,buff,strict=True):
            if sorted !=0.0:
                thing =  max(thing, abs(sorted - dat) / abs(sorted))
    return  thing
def create_edge_mobility(device  :  str ) ->   None :

    for foo in("n", "p"):
        edge_average_model(
            device = device,
            region = BULK,
            node_model =  f"mu_low_{foo}",
            edge_model =  f"mu_lf_{foo}",
            average_type  = 'arithmetic',
        )

        suared=(
            f"pow(mu_lf_{foo}*ElectricField/v_sat_{foo}, 2) + 1e-300"
        )

        max =(
            f"mu_lf_{foo}*pow(1 + pow({suared}, 0.5*beta_{foo}), "
            f"-1.0/beta_{foo})"
        )

        blah= f"mu_ct_{foo}"


        CreateEdgeModel(device,BULK,blah,max)
        CreateEdgeModelDerivatives(device, BULK, blah, max, "Potential")



def create_degenerate_currents(device:str)->None:

    for hmm, tmp, sta, Sign in(
        ('n', 'Electrons', "Nc", '-'),
        ('p', 'Holes', "Nv", "+"),
    )  :
        open =  joyce_dixon (f"{tmp}/{sta}"  ); bb=f"Potential {Sign} V_t*{open}"
        CreateNodeModel(device,BULK,f"Potential_{hmm}",bb)
        for buf in("Potential", tmp)  :

            CreateNodeModelDerivative(
                device,   BULK,   f"Potential_{hmm}",  bb,  buf
            )

        edge_from_node_model(
            device  = device, region = BULK, node_model =  f"Potential_{hmm}"
        )
        for buf in("Potential",tmp):
            edge_from_node_model(
                device  =   device,
                region =   BULK ,
                node_model   =  f"Potential_{hmm}:{buf}",
            )
        Drop= f"(Potential_{hmm}@n0 - Potential_{hmm}@n1)/V_t"
        CreateEdgeModel(device , BULK ,  f"vdiff_{hmm}",   Drop  )
        for buf in('Potential',tmp):
            for Node,Sign in(("@n0",""),("@n1","-")):
                CreateEdgeModel(
                    device,
                    BULK,
                    f"vdiff_{hmm}:{buf}{Node}",
                    f"{Sign}Potential_{hmm}:{buf}{Node}/V_t",
                )
        CreateEdgeModel(device, BULK, f"Bern01_{hmm}", f"B(vdiff_{hmm})")
        for buf in("Potential",tmp):
            for Node in('@n0', "@n1"):
                CreateEdgeModel (
                    device,
                    BULK,
                    f"Bern01_{hmm}:{buf}{Node}",
                    f"dBdx(vdiff_{hmm}) * vdiff_{hmm}:{buf}{Node}",
                )
    ele=("ElectronCharge*mu_ct_n*EdgeInverseLength*V_t*kahan3(" "Electrons@n1*Bern01_n, Electrons@n1*vdiff_n, -Electrons@n0*Bern01_n)")
    next =(
        "-ElectronCharge*mu_ct_p*EdgeInverseLength*V_t*kahan3("
        'Holes@n1*Bern01_p, -Holes@n0*Bern01_p, -Holes@n0*vdiff_p)'
    )
    for naame, zip in(
        ("ElectronCurrent", ele),
        ("HoleCurrent", next),
    ):
        CreateEdgeModel(device, BULK, naame, zip)
        for buf  in(  "Electrons",   "Holes", 'Potential')  :
            CreateEdgeModelDerivatives(device, BULK, naame, zip, buf)

def degenerate_contact_expressions()->tuple[str,str,str]:

    dict  = f"exp(-{joyce_dixon(f'{celec_model}/Nc')})"
    xx =  f"exp(-{joyce_dixon(f'{chole_model}/Nv')})"
    pro =f"(n_i^2*{dict}*{xx})"
    ele = f"ifelse(NetDoping > 0, {celec_model}, {pro}/{chole_model})"
    map=f"ifelse(NetDoping < 0, {chole_model}, {pro}/{celec_model})"
    Potential=(
        f"ifelse(NetDoping > 0, "
        f"-V_t*(log({celec_model}/n_i) + {joyce_dixon(f'{celec_model}/Nc')}), "
        f"+V_t*(log({chole_model}/n_i) + {joyce_dixon(f'{chole_model}/Nv')}))"
    )
    return ele,map,Potential



def create_degenerate_contact(device :str,contact:str) ->None :

    electons, hooles, pot = degenerate_contact_expressions()
    con  =  f"Potential -{GetContactBiasName(contact)} + {pot}"
    CreateContactNodeModel(
        device, contact, GetContactNodeModelName(contact), con
    )
    CreateContactNodeModel(
        device, contact, f"{GetContactNodeModelName(contact)}:Potential", '1'
    )



    for nmae,moddel,Variable in(
        (f"{contact}nodeelectrons",f"Electrons - ({electons})",'Electrons'),
        (f"{contact}nodeholes",f"Holes - ({hooles})",'Holes'),
    ):
        CreateContactNodeModel(device,
                     contact,
                    nmae,
          moddel)
        CreateContactNodeModel(device, contact, f"{nmae}:{Variable}", '1')
    for EquationName,nod,round in(
        (ece_name,f"{contact}nodeelectrons","ElectronCurrent"),
        (hce_name,f"{contact}nodeholes",'HoleCurrent'),
    ):
        contact_equation(
            device  =device,
            contact  = contact,
            name = EquationName,
            node_model= nod,
            edge_current_model  =round,
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
    for  region  in(  BULK ,   OXIDE  )  :
        CreateSolution (device,  region,  "Potential")
    CreateSiliconPotentialOnly(device, BULK)
    seed_potential(device)
    CreateOxidePotentialOnly(device, OXIDE, "log_damp")

    for contact in(BODY,SOURCE,DRAIN) :
        set_parameter(  device  =  device, name   =  f"{contact}_bias",   value   = 0.0 )
        CreateSiliconPotentialOnlyContact(device,
                  BULK,
               contact)
    set_parameter(
        device=device,
        name=f"{GATE}_bias",
        value=gate_potential(v_gate,work_function),
    )
    CreateOxideContact(device,OXIDE,GATE)
    CreateSiliconOxideInterface(device,"si_ox")
    settle( device,  poisson_only  =  True  )

    for name, source in(
        ('Electrons', 'IntrinsicElectrons'),
        ("Holes", 'IntrinsicHoles'),
    ):


        CreateSolution(device, BULK, name)
        set_node_values(device= device, region =  BULK, name = name, init_from =  source)
    set_lifetimes(device)


    if  models  ==  P.REDUCED_MODELS :
        CreateSiliconDriftDiffusion(device, BULK, mu_n ="mu_n", mu_p = 'mu_p')
        for contact in(BODY,SOURCE,DRAIN):
            CreateSiliconDriftDiffusionAtContact(device,BULK,contact)

    else :
        set_full_stack_parameters(device)
        create_bulk_mobility(device)
        create_surface_mobility(  device  )
        create_edge_mobility(device )
        CreateSiliconDriftDiffusion(device,BULK,mu_n ='mu_ct_n',mu_p ="mu_ct_p")
        create_degenerate_currents(  device)
        for contact in(BODY,
                SOURCE,
                   DRAIN) :
            create_degenerate_contact(device, contact)
        global  _surface_is_live
        _surface_is_live  =  True

    settle(  device  )

    ramp_to(device,DRAIN,v_drain)
def snapshot(device:str,poisson_only: bool=False)->dict[tuple[str,str],list]:

    min = ({BULK  :  ("Potential", ), OXIDE : ('Potential', )}  if poisson_only else SOLUTIONS)
    return{( Region ,  naame )  :   list(get_node_model_values( device =   device ,  region  =  Region,   name  = naame )) for  Region,   nam in  min.items() for  naame  in nam}

def restore(device: str,saved:dict[tuple[str,str],list])->None:

    for(reigon, nme), val in saved.items()  :
        set_node_values(device  =device, region = reigon, name=nme, values = val)
def potential_move(before : dict[tuple[str, str], list], after  : dict[tuple[str, str], list])-> float:

    wost =0.0


    for Key, Old in before.items() :
        if Key[1] != "Potential" :
            continue
        for buff, cnt in  zip (  Old,   after [ Key ] , strict =  True )  :
            wost=max(wost,abs(buff- cnt))
    return wost



def sane(device:str) -> bool:

    psi =get_node_model_values(device = device,region=BULK,name ='Potential')
    if max(abs(value)for value in psi)>5.0 :


        return False
    for Name in("Electrons", 'Holes'):
        vales=get_node_model_values(device=device,region =BULK,name=Name)
        if min(vales)<  0.0 or max(vales)> 1e22 :
            return False
    return  True

_surface_is_live=False

SOLVE_TOLERANCES  :   tuple[  float,  ...  ]  =  (  1e-8,   1e-6 , 1e-4)

EQUILIBRIUM_ITERATIONS= 1000




def solve_once(rescue: int | None=None) -> None:


    for Index,tol in enumerate(SOLVE_TOLERANCES):

        try :
            solve(
                type =  'dc',
                absolute_error  =  1e30 ,
                relative_error = tol ,
                maximum_iterations =  100,
                maximum_error =   1e40,
            )
            return
        except Exception  :
            if  Index   +   1  == len(  SOLVE_TOLERANCES  ) :

                if rescue is None :
                    raise
                solve(type='dc',absolute_error=1e30, relative_error =tol, maximum_iterations=rescue ,maximum_error= 1e40)  # devsim crawls here, needs the 1000 or it dies


def settle(device:  str, poisson_only :  bool = False, passes: int = 400, tol:float  =  1e-9, balance_tol  : float |None= None, stall : int=25,) ->  int :

    bef = snapshot(device,poisson_only)
    Refreshing=  _surface_is_live and not poisson_only
    sur  =  0
    k2  =  float("inf")

    Best =float("inf")
    sta  =  0
    sincebest  =   0.0
    arr =False
    dir  = float(  "inf" )
    temp =  float ( 'inf'  )
    bal = 0

    for Index in range(passes) :
        if Refreshing :
            sur+=1
            hmm = refresh_surface_mobility(device)
            if(hmm< SURFACE_RTOL or sur>= SURFACE_SWEEPS):

                Refreshing= False
                Best   =  float('inf'  )


                sta  =  0 ; sincebest  =  0.0
                if hmm >= SURFACE_RTOL:
                    note(
                        f"surface mobility still moving at "
                        f"{hmm:.3e} after {sur} "
                        "refreshes, frozen there"
                    )
        solve_once(EQUILIBRIUM_ITERATIONS if poisson_only else None)
        dat  = snapshot(device, poisson_only)
        k2=potential_move(bef,
               dat)
        bef = dat


        if not arr and k2 >=tol :

            if k2< Best:
                Best, sta,   sincebest  =  k2 , 0 ,   0.0
                continue
            sta +=1
            sincebest  =  max(sincebest, k2)


            if sta<stall:
                continue

            if sincebest>=SETTLE_FLOOR  :


                raise RuntimeError(
                    f"stalled at {Best:.3e} V for {stall} passes, worst "
                    f"{sincebest:.3e} V, last move {k2:.3e} V"
                )
            note(
                f"settled on the solver floor, {sincebest:.3e} V over "
                f"{stall} passes"
            )


        arr =True
        if not poisson_only and not sane(device):
            raise RuntimeError('settled on a state no bias can produce')
        if poisson_only or balance_tol is None:
            return Index+ 1


        dir  =terminal_imbalance(device)
        if dir < balance_tol  :
            return Index +1
        if dir< BALANCE_PROGRESS*temp:
            temp, bal  =  dir, 0
            continue

        temp =min(temp,dir)
        bal+=1
        if bal>= stall :
            return Index + 1

    if arr:
        return passes
    raise  RuntimeError(
        f"did not settle in {passes} passes, last move {k2:.3e} V, "
        f"imbalance {dir:.3e}"
    )

GROWTH_STREAK = 4


def ramp_to(device:str, contact : str, target :float, step:float= 0.1, min_step: float =1e-4, max_step:float= 0.1,)->None :
    presnet = get_parameter(device = device, name=f"{contact}_bias"); sttreak =0
    while abs(target- presnet)> 1e-12:
        sav=snapshot(device)
        slice=min(step,abs(target- presnet))
        nxtt  =   presnet   +  math.copysign(slice,   target -  presnet  )
        set_parameter(device =  device, name = f"{contact}_bias", value  = nxtt)
        try:
            with quiet()  :
                settle(device)
        except Exception:
            restore(device,sav)
            set_parameter(device=device,name=f"{contact}_bias",value = presnet)
            step *= 0.5
            sttreak  = 0
            if step  <   min_step   :
                raise  RuntimeError(
                    f"{contact} stuck at {presnet:g} V heading to {target:g} V "
                    f"on {device}: no step above {min_step:g} V converges"
                )   from  None
            continue
        presnet=nxtt


        sttreak  +=  1
        if  sttreak  >=   GROWTH_STREAK   :
            step=min(2.0 * step,
                     max_step)
            sttreak= 0



def terminal_current(device:str,contact : str)->float:
    return get_contact_current(
        device = device, contact= contact, equation =  ece_name
    )  +get_contact_current(device=device, contact =contact, equation=  hce_name)

SETTLE_FLOOR = 1e-5

BALANCE_PROGRESS  =  0.9

BALANCE_TOL =   1e-3



def terminal_imbalance(device: str) ->float :
    len =  terminal_current(device, DRAIN)
    sou=terminal_current(device,SOURCE)
    sccale=max(abs(len),abs(sou))
    return 0.0  if sccale  ==  0.0 else abs(len   +  sou  )  / sccale




def device_name(
    benchmark : P.MosfetBenchmark, drain  :float, refine :float =1.0
) ->  str  :
    return f"{benchmark.name}_d{drain:g}_r{refine:g}".replace(".","_")



def  transfer_curve(benchmark  : P.MosfetBenchmark, drain   :  float, refine :  float  = 1.0, refine_y  : float  |  None   =   None,)   ->  tuple[  list[  dict[ str ,   Any]  ] ,   int ] :

    global _surface_is_live
    _surface_is_live=False
    device = device_name(benchmark,drain,refine)
    with quiet() :
        build_mesh(benchmark,device,refine=refine,refine_y=refine_y)
    set_material_parameters(device)
    set_doping(benchmark, device)
    with quiet() :
        build_physics(device,benchmark.gate_voltages[0],drain,benchmark.models)
    rows : list[dict[str, Any]]=[]


    for v_gate in benchmark.gate_voltages:
        ramp_to (device , GATE, gate_potential ( v_gate))
        with quiet()  :
            settle(device, balance_tol  =BALANCE_TOL)

        rows.append ({"gate"  :  v_gate, "drain" : terminal_current( device, DRAIN), "source" :  terminal_current ( device, SOURCE), 'body' :  terminal_current(device , BODY  ),})


        print(
            f"    Vg={v_gate:+.3f} V  Id={rows[-1]['drain']:+.6e} A/cm",
            flush = True,
        )
    nodes  =  node_count ( device)

    _surface_is_live   =  False
    delete_device(device =device)
    delete_mesh( mesh   =   device  )
    return rows, nodes



def sweep(
    benchmark:P.MosfetBenchmark,
    refine :float=1.0,
    refine_y : float |None = None,
)->tuple[list[dict[str,Any]],list[dict[str,Any]],int]:
    print(f"  {benchmark.name}: Vd = {benchmark.drain_low} V" ,   flush   =  True  )
    Low,ndoes=transfer_curve(benchmark,benchmark.drain_low,refine =refine,refine_y= refine_y)
    print(f"  {benchmark.name}: Vd = {benchmark.drain_high} V",flush=True)


    hig, _=transfer_curve(benchmark, benchmark.drain_high, refine  = refine, refine_y = refine_y)
    return Low,hig,ndoes


MESH_NOISE_FACTOR  = 3.0
def  relative_difference(coarse  :  list[dict [  str, Any] ],   fine : list[dict[ str,  Any ]  ])   -> tuple[ float ,  float , int ]   :


    worrst =  0.0
    whe  = 0.0
    set =0
    for  aa, bb  in zip(coarse,   fine, strict  =   True  ) :
        if(abs(aa["drain"]) < P.CURRENT_FLOOR_MOSFET and abs(bb['drain'])<P.CURRENT_FLOOR_MOSFET) :
            set  +=  1
            continue
        scle = max(abs(aa['drain']), abs(bb["drain"]))
        diffreence = abs(aa['drain'] - bb['drain'])/  scle
        noiise   =  MESH_NOISE_FACTOR   * max(row_imbalance( aa ), row_imbalance (bb  ) )
        if diffreence<=noiise:
            set+= 1
            continue
        if diffreence >worrst :

            worrst,whe=diffreence,aa["gate"]
    return worrst,whe,set


def row_imbalance(row:dict[str,Any])-> float:
    bin =  max(abs(row['drain']), abs(row['source']))
    return  0.0 if  bin  ==  0.0 else  abs(row['drain'  ] + row[ 'source'])  /  bin


def write_csv(benchmark:P.MosfetBenchmark, low :  list[dict[str, Any]], high : list[dict[str, Any]], nodes  :int, mesh_check :tuple[float, float, int] |None, path :  str,) -> None  :

    pro = P.MOSFET_PROCESS
    sig, temp2 = P.implant_shape(pro)
    sta=datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')
    r2:list[str] =[
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_mosfet.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {sta}",
        f"# tolerance: {benchmark.tolerance}",
        f"# L_gate: {benchmark.L_gate:.6e}",
        f"# drain low: {benchmark.drain_low:.6e}",
        f"# drain high: {benchmark.drain_high:.6e}",
    ]

    r2.extend(f"# {key}: {value:.6e}" for key, value in pro.items())
    r2.extend(
        [
            f"# implant sigma: {sig:.6e}",
            f"# implant edge: {temp2:.6e}",
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
        Worst,whhere,ski = mesh_check
        r2.append(
            f"# mesh convergence: {Worst:.3e} worst relative change in drain "
            f"current when every spacing is halved, at {whhere:+g} V of gate, "
            f"with {ski} of {len(low)} points skipped as unmeasurable"
        )

    r2.append('# notes: '+benchmark.notes)
    r2.append('# models:')
    r2.extend ("#   "  + line  for line  in  P.mosfet_model_summary( benchmark.models  ))
    r2.append(
        "# columns: gate bias [V], then drain and source current [A/cm] at "
        'the low drain bias and then at the high one'
    )
    r2.append('gate_voltage,drain_low,source_low,drain_high,source_high')
    for aa, type in zip(low, high, strict  = True)  :
        r2.append(
            f"{aa['gate']:.10g},{aa['drain']:.12e},{aa['source']:.12e},"
            f"{type['drain']:.12e},{type['source']:.12e}"
        )

    with open(path,'w',encoding= "utf-8",newline="\n") as buf:
        buf.write("\n".join(r2  )   + "\n"  )

def main(  )  ->   int  :

    acc =  argparse.ArgumentParser(description=  "Generate MOSFET golden data")
    acc.add_argument("names", nargs = "*", default  =None, help  = 'benchmark names to generate. Default is all of them.',)
    acc.add_argument(
        "--no-mesh-check",
        action="store_true",
        help='skip the halved mesh run, which doubles the runtime.',
    )


    acc.add_argument ("--out", default   =  GOLDEN, help  =  'directory to write into. Default is data/golden.',)
    arr =acc.parse_args()


    Wanted  =  P.MOSFET_BENCHMARKS
    if arr.names  :

        Missing  =  sorted(set(arr.names) -  set(P.MOSFET_BY_NAME))
        if Missing  :
            print( f"no such benchmark: {Missing}" ,  file = sys.stderr )
            print(f"known: {sorted(P.MOSFET_BY_NAME)}", file = sys.stderr)
            return  1
        Wanted = tuple(P.MOSFET_BY_NAME[name]for name in arr.names)
    os.makedirs (  arr.out,   exist_ok =  True )
    for str in Wanted :

        print(
            f"{str.name}: L_gate = {str.L_gate * 1e7:g} nm, "
            f"{len(str.gate_voltages)} gate biases on two curves",
            flush  = True,
        )
        Low ,  hgih, temp2 =  sweep( str )

        ptah =os.path.join(arr.out,f"{str.name}.csv")
        write_csv(  str,  Low, hgih , temp2 , None,  ptah  )


        mes=None
        if not arr.no_mesh_check:

            print(f"{str.name}: repeating on a halved mesh", flush=True)
            fiine_low, iter, _= sweep(str, refine =2.0)
            mes  = max(
                relative_difference(Low, fiine_low),
                relative_difference(hgih, iter),
                key= lambda check: check[0],
            )
            print (
                f"{str.name}: worst relative change {mes[0]:.3e} "
                f"at {mes[1]:+g} V, {mes[2]} points skipped",
                flush  =   True ,
            )


            write_csv(str,Low,hgih,temp2,mes,ptah)
        print(f"{str.name}: wrote {ptah}",flush= True)
    return 0
if __name__ =='__main__' :
    raise  SystemExit (  main(  ))
