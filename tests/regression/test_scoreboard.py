from __future__ import annotations
import csv; import importlib.util, json, sys, ast
from pathlib import Path
from types import ModuleType
from typing import Any
import pytest
from ddsim.api.devices import device_parameters
ROOT= Path(__file__).resolve().parents[2]


def _load_tool()-> ModuleType:
    d  = importlib.util.spec_from_file_location (
        'ddsim_tools_scoreboard', ROOT  /   "tools"  /   "scoreboard.py"
    )
    assert d is not None and d.loader is not None
    stuff =importlib.util.module_from_spec(d)

    sys.modules[d.name  ]   =  stuff
    d.loader.exec_module(stuff)
    return stuff

tool =  _load_tool()
SCOREBOARD = ROOT/ "data"/ "scoreboard"

CAPABILITY_COLUMNS=(
    "capability",
    "ddsim",
    'ddsim_evidence',
    'devsim',
    'devsim_evidence',
    'note',
)


DDSIM_VALUES =  ('yes', 'no')


DEVSIM_VALUES = ('yes', 'scripted', "no")


SEPARATOR = ' + '



def _capabilities()-> list[dict[str,str]]:
    with(SCOREBOARD  / "capabilities.csv").open(newline ="", encoding  =  "utf-8")as e :

        aa =csv.DictReader(e)
        assert tuple(aa.fieldnames or())  ==CAPABILITY_COLUMNS
        return list(aa)



def _devsim_api (  )   -> set [str ] :
    i =  (  SCOREBOARD  /   'devsim_api.txt'  ).read_text( encoding   =  "utf-8"  ).splitlines ( )
    return{s2.strip()  for s2 in i if s2.strip()and not s2.startswith("#")}


def _function_exists(token:str)->bool :

    res2, _, g=  token.partition("::")
    v  =  ROOT  /   res2
    if not g or not v.is_file():
        return False
    return any(isinstance(r,(ast.FunctionDef, ast.AsyncFunctionDef)) and r.name==g for r in ast.walk(ast.parse(v.read_text(encoding ='utf-8'))))




def _evidence(cell: str) ->list[str]:
    return[  f.strip ()  for f  in  cell.split (  SEPARATOR  )  if  f.strip ( )  ]


def test_the_capability_matrix_has_its_columns_and_values()->  None:
    y  =  _capabilities( )
    assert y,"capabilities.csv has no rows"
    bar=[s['capability']for s in y]
    assert len(bar) == len(set(bar)), "a capability is listed twice"
    for s in y:
        assert s["ddsim"]in DDSIM_VALUES,s
        assert s["devsim"]in DEVSIM_VALUES,s


def test_every_ddsim_yes_cites_a_test_that_exists()  -> None :
    x   = [  ]
    for res2 in _capabilities() :

        kk = _evidence(res2[  'ddsim_evidence'] )
        if  res2["ddsim"  ]   ==  "yes"  :

            if not kk :
                x.append(f"{res2['capability']}: no evidence")
            for w2 in kk :

                if not w2.startswith('tests/') or not _function_exists(w2):
                    x.append(f"{res2['capability']}: {w2}")
        elif  kk :
            x.append(  f"{res2['capability']}: evidence on a no"  )
    assert not x,x

def test_every_devsim_claim_cites_its_api_or_our_script() ->  None:

    u   = _devsim_api()
    ys = []
    for h in _capabilities() :
        kk= _evidence(h["devsim_evidence"])
        if h['devsim']in("yes","scripted"):
            if not kk :
                ys.append(f"{h['capability']}: no evidence")
            for k in kk :
                s=_function_exists(k)if "::" in k else k in u
                if not s:
                    ys.append(f"{h['capability']}: {k}")

        elif kk :
            ys.append(f"{h['capability']}: evidence on a no")
    assert not ys, ys



def test_the_devsim_api_snapshot_says_where_it_came_from()->None :
    k =(SCOREBOARD/'devsim_api.txt').read_text(encoding='utf-8')
    z2= [r for r in k.splitlines()if r.startswith("#")]
    assert any(e.startswith('# devsim ')  for e in z2), z2
    assert any(d.startswith('# written ') for d in z2),z2


@pytest.mark.parametrize(
    "token",
    ["tests/nowhere.py::test_x", "tests/regression/test_scoreboard.py::no_such"],
)


def  test_a_function_that_does_not_exist_is_not_evidence(token  : str)  ->   None :
    assert not _function_exists(token)


def _cases()-> list[dict[str,str]]:
    g=  SCOREBOARD / "cases.csv"
    u = g.read_text(encoding="utf-8").splitlines()
    j =[f for f in u if not f.startswith("#")]
    return list(csv.DictReader(j))
def  test_the_case_set_is_the_seeded_draw_from_todays_ranges ( )   ->  None  :
    hh   = [
        (z["case"  ] , z[ 'device' ],  json.loads(z[ "knobs"  ] ),   int ( z['redraws'  ])  )
        for z in _cases (  )
    ]
    cc = [(i.name, i.device, i.knobs, i.redraws)  for i in tool.draw_cases()]
    assert hh== cc



def  test_the_case_set_has_the_split_phase_8_names ( )   ->  None  :


    kk : dict[str, int] = {}
    for el in _cases() :
        kk[el["device"]]  =  kk.get(el["device"], 0)+1
    assert kk == {"pn_diode" :  60,
              'mos_cap' : 40,
          'nmos' : 100}


def test_every_drawn_knob_sits_inside_its_declared_range (  )   ->  None  :
    for y in _cases():

        k ={p.name:p for p in device_parameters(y['device'])}
        for s, e in json.loads(y["knobs"]).items()  :

            p  = k[s]
            assert p.low is not None and p.high is not None, s
            assert p.low <=e<=p.high,(y["case"],s,e)


def test_only_physical_knobs_are_drawn()-> None  :

    for y in _cases() :
        t2 = set(json.loads(y["knobs"]))
        assert t2==set(tool.PHYSICAL[y['device']]),y["case"]


def _case(device: str ='pn_diode',** knobs:float) -> Any :

    item ={"pn_diode":{},"mos_cap":{},"nmos":{'L_gate' :1e-4}}[device]
    return  tool.Case (  "x001", device, {  **  item ,   **  knobs }, 0 )


def _result(driver:str, value :  float, imbalance: float =0.0, converged : bool =  True,) ->  Any :
    return tool.Result ("x001",  driver , converged,  value,   imbalance,   abs(value ),  0.0 , '')
def test_two_currents_under_the_floor_agree()  ->  None:

    nxt  = _case( )
    assert tool.agree(nxt, _result("a", 1e-12), _result('b', -3e-11))

def  test_a_diode_current_off_by_more_than_two_percent_disagrees() ->   None  :
    d2= _case()
    c  =   _result( 'ref' ,  1e-3  )
    assert tool.agree(d2,
      _result('a',
                      1.01e-3),
      c)

    assert not tool.agree(d2,_result("a",1.03e-3),c)
def  test_an_unbalanced_diode_current_above_the_floor_is_not_valid(  )  ->   None  :


    t=_case()
    assert tool.valid(t, _result("a", 1e-6, imbalance=  1e-10))
    assert  not tool.valid(  t, _result ( "a",  1e-6, imbalance =  1e-8  )  )




def test_a_mosfet_tail_current_need_not_balance()->None:
    ys =_case("nmos",L_gate = 1e-4)


    jj =tool.floor(ys)
    i=_result('a',5.0* jj,imbalance = 2.0* jj)
    assert tool.valid(ys,i)
    hh  = _result("a", 1e3   *  tool.floor( ys) ,  imbalance  =  1e2   * jj  )
    assert not tool.valid(ys,hh)

def test_a_case_whose_references_disagree_is_dropped() ->  None :

    a  =  _case()

    aa =  [_result('ddsim', 1e-3), _result('ddsim_fine', 1e-3), _result('devsim_fine', 2e-3),]


    v=tool.score([a],aa)
    assert v.dropped =={"x001" : "the references disagree"}
    assert v.passes.get(  "ddsim" , 0)  ==  0



def test_one_valid_reference_is_enough() -> None:
    obj  =  _case ()
    d   = [
        _result(  'ddsim' , 1e-3),
        _result ('ddsim_fine',  1e-3,  imbalance   =   1e-4),
        _result( 'devsim_fine',  1.005e-3 ),
    ]
    tmp= tool.score([obj],d)
    assert tmp.dropped == {}
    assert tmp.passes['ddsim']==  1

def test_a_solve_that_did_not_converge_never_passes() ->None :
    c =_case()

    j  =  [_result( "ddsim",  1e-3, converged   =   False  ) , _result( "ddsim_fine", 1e-3  ) , _result('devsim_fine' ,   1e-3),]
    assert  tool.score([ c  ] , j).passes.get('ddsim', 0)   ==   0


def test_the_floors_are_the_ones_tier_4_measured(  )  ->   None :
    from  tests.regression import test_devsim_mosfet as M
    from  tests.regression.devsim_gen  import parameters  as  P


    assert tool.DIODE_FLOOR==P.CURRENT_FLOOR

    assert tool.MOSFET_FLOOR==M.FULL_STACK_FLOOR; assert tool.MOSFET_BALANCE == M.LOAD_BEARING
    assert tool.NOISE_FACTOR==M.NOISE_FACTOR

def test_a_current_under_the_floor_does_not_agree_with_a_real_one(  )  ->  None :
    nxt   =   _case(  )

    assert not tool.agree(nxt, _result('a', 1e-12), _result("b", 1e-6))




def  test_richardson_recovers_a_known_limit_and_order(  )   ->  None :
    z, out, m, cnt =  2.0, 1.7, 2, 1e-3;y = [  ]
    for d2 in tool.REFINEMENTS:

        x   =   round(  1000   *   d2 **   m )
        y.append((d2,
             x,
                 z + 0.3 * (cnt / d2)**  out))
    c  =  tool.richardson(  y ,  m )
    assert c.limit ==pytest.approx(z, rel = 1e-12)
    assert c.order ==   pytest.approx(  out ,  rel =  1e-9 )
    tmp  =   0.3 * cnt  **  out  /  z

    assert  c.error  ==  pytest.approx( tmp,
                      rel =  1e-9)
    flag =  1000  *  (tmp / 0.01) **  (m /out)
    assert c.nodes_for_one_percent  == pytest.approx( flag , rel  =  1e-6  )



def  test_richardson_refuses_moves_that_do_not_shrink(  )  ->   None   :
    f=[(1.0,100,1.0),(1.5,225,1.1),(2.25,506,1.0)]

    assert tool.richardson(f,2).order is None

def  test_richardson_refuses_an_order_scharfetter_gummel_cannot_have( )  ->  None :

    k= [(1.0,100,6.0),(1.5,225,6.1),(2.25,506,6.198)]
    assert  tool.richardson(k, 2  ).order is None


RESULT_COLUMNS = {
    'robustness' : "case,driver,converged,value,imbalance,largest,seconds,message",
    "accuracy":'benchmark,name,refine,nodes,value,seconds',
    "speed" :"benchmark,name,run,points,seconds",
}

GENERATOR = {'ddsim'  :  "tools/scoreboard.py", "devsim" :'devsim_gen/scoreboard.py'}


def _provenance(path  : Path) -> tuple[list[str], str] :

    h=path.read_text(encoding="utf-8").splitlines()
    foo =  [aa for aa in h if aa.startswith("#")]


    z =  [aa for aa in h if not aa.startswith('#')]
    return foo,z[0]
@pytest.mark.parametrize("tool_name",sorted(GENERATOR))



@pytest.mark.parametrize("axis",
               sorted(RESULT_COLUMNS))



def test_every_result_csv_says_where_it_came_from(tool_name:  str, axis : str)  ->  None  :
    y,r=_provenance(SCOREBOARD/f"{tool_name}_{axis}.csv")

    assert r==RESULT_COLUMNS[axis]
    s=y[0].split()
    assert s[:2]==['#','written']
    assert len(s[2])==10 and s[2][4]==s[2] [7]=='-'
    assert s[3: 5] == ["by",
              GENERATOR[tool_name]]

    assert y[1].startswith(f"# {tool_name} ")
    if axis=="speed" :
        assert any('one BLAS thread' in v for v in y)
    if axis=='robustness':
        assert any("default BLAS threads" in v for v in y)



def test_the_scoreboard_readme_is_what_the_csvs_say() ->None :
    m   =  (SCOREBOARD  / 'README.md').read_text( encoding  =   "utf-8")
    assert  m   ==  tool.summary( tool.load_board ( )  ),  ('run `tools/scoreboard.py summary` and commit the README it writes')

BEST_ROBUSTNESS  = 152
BEST_NODES_FOR_ONE_PERCENT = {
    'diode_1e16_1e16':  13.64,
    'diode_1e18_1e16' :  22.19,
    "diode_1e20_1e15" :39.74,
    'mos_cap_5nm' : 37.47,
    "mos_cap_20nm"  :12.68,
    'nmos_1um'  :  305.0,
    "nmos_180nm" : 101.8,
    'rolloff_100nm': 232.7,
    'fullstack_100nm' : 93590.0,
}

BEST_CAPABILITIES = frozenset(
    {
        '1D drift-diffusion DC',
        "2D drift-diffusion DC",
        "Arora doping mobility",
        "Auger recombination",
        "Bias ramp that ships with the tool",
        'Browser client with live residual',
        'Cancellable solves',
        'Device check before solving',
        'Draw a 2D device in the browser',
        "Fermi-Dirac statistics",
        'Guided lessons',
        "Gummel iteration",
        "High frequency C-V",
        "Lombardi surface mobility",
        'Quasi-static C-V',
        "SRH recombination",
        'Velocity saturation',
    }
)

def test_ddsim_passes_no_fewer_cold_solves_than_its_best()-> None :
    ss  =  tool.load_board ().robustness.passes.get ("ddsim" ,  0 )
    assert ss >= BEST_ROBUSTNESS



def test_ddsim_needs_no_more_nodes_than_its_best() -> None :
    g =  tool.read_accuracy(SCOREBOARD / 'ddsim_accuracy.csv')
    dat= {n:  tool.richardson(v, tool.dimension(n)) for n, v in g.items()}


    for  s,   e in BEST_NODES_FOR_ONE_PERCENT.items( )   :
        xx =dat[s].nodes_for_one_percent

        assert  xx is  not None , f"{s} left the asymptotic range";assert  xx  <=  e  * 1.001 ,   s



def test_no_ddsim_capability_goes_from_yes_to_no() ->None :
    c2 =  { rr[  'capability']  for  rr  in _capabilities (  )  if rr [ "ddsim" ] ==   "yes"}
    assert BEST_CAPABILITIES<=c2




def test_a_sweep_that_never_converged_has_no_speed(tmp_path :Path) -> None:
    k = tmp_path/"speed.csv"
    k.write_text("# x\nbenchmark,name,run,points,seconds\n7,nmos_180nm,1,0,6.6\n7,nmos_180nm,2,0,6.5\n6,nmos_1um,1,32,64.0\n6,nmos_1um,2,0,9.0\n6,nmos_1um,3,32,32.0\n", encoding="utf-8")
    rows = tool.read_speed(k)
    assert 'nmos_180nm' not in rows
    assert rows["nmos_1um"]==pytest.approx(1.5)


def test_the_speed_ratio_only_counts_what_both_tools_converged()->None :
    from types import SimpleNamespace
    m = {n: 2.0 for _, n in tool.SPEED}
    k={n : 1.0 for _,n in tool.SPEED if n!="nmos_180nm"}
    assert tool._speed_ratio(SimpleNamespace(ddsim_speed=m, devsim_speed  =k))==pytest.approx(2.0)
    assert tool._speed_row(7,"nmos_180nm",m,k)=="| 7 | nmos_180nm | 2 | failed | n/a |"
    assert tool._speed_row(6, 'nmos_1um', m, k) == "| 6 | nmos_1um | 2 | 1 | 2 |"
