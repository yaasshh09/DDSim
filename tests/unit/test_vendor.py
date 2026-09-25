"""What the page loads besides its own scripts, vendored.

phases/PHASE-7.md part two: nothing loads from a CDN, the page works with no
network. So marked, KaTeX and the two fonts the design is set in live in the
repo, and this file holds them to the manifest tools/vendor_client_libs.py
wrote: every file present, every hash matching, every licence shipped. A hand
edit to a vendored file fails here.
"""

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
    """The page itself is checked below, but a stylesheet it links can fetch
    just as easily, and the font css arrives from Google full of urls at
    fonts.gstatic.com. Rewriting every one of them to a local file is the
    whole point of vendoring the fonts rather than linking them."""
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
