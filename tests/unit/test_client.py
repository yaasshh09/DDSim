from __future__ import annotations


import re
import pytest
from ddsim.api.app import PAGE
from ddsim.core import constants

JS = PAGE.parent  /'js'



def first_party_script()-> str:
    return "\n".join(
        path.read_text(encoding  =  "utf-8")  for path in sorted(JS.glob('*.js'))
    )
CLIENT=PAGE.read_text(encoding = "utf-8")+  first_party_script()




def  constant_names (  )   -> list[ str]   :
    return[
        nam
        for nam in dir(constants)
        if not nam.startswith("_") and nam not in{'annotations','np'}
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
    scipt  = first_party_script()
    assert scipt.count('Math.log10') == 1
    assert scipt.count("Math.pow")==  1
    assert "const decades = (v) => Math.log10(v);" in scipt
    assert "const undecades = (v) => Math.pow(10, v);" in  scipt
def test_no_silicon_constant_is_named_in_the_client()-> None :
    buff  = first_party_script()
    tmp = [
        nmae
        for nmae in constant_names()
        if re.search(rf"\b{re.escape(nmae)}\b", buff)
    ]


    assert tmp== [],f"the client mentions {tmp}"


def test_the_client_does_not_convert_between_scaled_and_physical() ->None :
    Script  =  first_party_script(  )

    for scalling in("V_T", '0.0259', '38.7', "kT", "thermal") :


        assert scalling not in Script


def test_streamline_tracing_uses_no_physical_function()  ->  None  :
    traacing= (JS/ "streamlines.js").read_text(encoding="utf-8")

    for  sum  in( 'Math.exp' ,  "Math.log" ,  'Math.pow',  "Math.sinh"  ) :
        assert  sum  not  in traacing
