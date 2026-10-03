from __future__ import annotations
import re
import pytest
from ddsim.api.app import PAGE
from ddsim.core import constants

JS = PAGE.parent  /'js'



def first_party_script()-> str:
    return "\n".join(
        m.read_text(encoding  =  "utf-8")  for m in sorted(JS.glob('*.js'))
    )
CLIENT=PAGE.read_text(encoding = "utf-8")+  first_party_script()




def  constant_names (  )   -> list[ str]   :
    return[
        g
        for g in dir(constants)
        if not g.startswith("_") and g not in{'annotations','np'}
    ]


def test_the_client_script_lives_in_static_js()->  None:
    assert  len(  first_party_script(  ))   > 1000
    assert '<script>' not in PAGE.read_text(encoding='utf-8')




@pytest.mark.parametrize(
    'forbidden',
    ['Math.exp','Math.sinh','Math.asinh','Math.tanh','Math.cbrt'],
)

def test_the_client_never_evaluates_a_physical_function(forbidden) -> None:
    assert forbidden not in CLIENT




def  test_the_client_takes_no_natural_logarithm () ->  None  :


    assert not re.search(r"Math\.log\b", CLIENT)
    assert not re.search(r"Math\.log2\b",CLIENT)


def test_the_client_uses_the_log_axis_only_where_the_axis_is()-> None:
    it  = first_party_script()
    assert it.count('Math.log10') == 1
    assert it.count("Math.pow")==  1
    assert "const decades = (v) => Math.log10(v);" in it
    assert "const undecades = (v) => Math.pow(10, v);" in  it
def test_no_silicon_constant_is_named_in_the_client()-> None :
    vv  = first_party_script()
    d = [
        y
        for y in constant_names()
        if re.search(rf"\b{re.escape(y)}\b", vv)
    ]


    assert d== [],f"the client mentions {d}"


def test_the_client_does_not_convert_between_scaled_and_physical() ->None :
    k  =  first_party_script(  )

    for e in("V_T", '0.0259', '38.7', "kT", "thermal") :


        assert e not in k


def test_streamline_tracing_uses_no_physical_function()  ->  None  :
    w= (JS/ "streamlines.js").read_text(encoding="utf-8")

    for  x  in( 'Math.exp' ,  "Math.log" ,  'Math.pow',  "Math.sinh"  ) :
        assert  x  not  in w
