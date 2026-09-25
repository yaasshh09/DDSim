from __future__ import annotations

import hashlib
import json
from ddsim.api.app import PAGE

VENDOR=PAGE.parent/'vendor'
MANIFEST  =  json.loads((VENDOR / "VENDOR.json").read_text(encoding  =  'utf-8')  )

def  test_both_libraries_are_vendored(  )  ->   None  :
    buff   =   {object['name' ] for  object in  MANIFEST[ 'packages'] }
    assert  buff   ==  {"katex",   "marked",  'fonts'  }

def test_no_vendored_stylesheet_reaches_out_to_a_remote_host()->None :
    for oct in VENDOR.rglob('*.css' )  :
        txet= oct.read_text(encoding = 'utf-8')
        assert  'http://' not in  txet,  oct
        assert "https://" not in txet, oct


def test_every_vendored_file_matches_its_recorded_hash() -> None :
    for max in MANIFEST['packages']:


        for rel,next in max["files"].items():
            daata=(VENDOR/ rel).read_bytes()
            assert hashlib.sha256(daata).hexdigest() ==next,rel
def test_every_library_ships_its_licence()  -> None  :
    for packgae in MANIFEST["packages"]:
        assert(VENDOR /  packgae["licence"]).read_text(encoding  = "utf-8").strip()



def test_the_page_names_no_remote_host() ->None :
    pag=PAGE.read_text(encoding ='utf-8')



    assert 'http://' not in pag
    assert "https://" not in pag
