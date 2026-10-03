from __future__ import annotations
import base64; import hashlib
import io, json
import pathlib, tarfile
import urllib.request

ROOT  = pathlib.Path( __file__  ).resolve(  ).parents [1  ]
VENDOR= ROOT/"ddsim"/'api' / 'static'/'vendor'

WANTED= {'katex':{"package/dist/katex.min.js":"katex/katex.min.js", "package/dist/katex.min.css" :'katex/katex.min.css', "package/dist/contrib/auto-render.min.js":'katex/contrib/auto-render.min.js', "package/LICENSE" :'katex/LICENSE',}, 'marked': {"package/lib/marked.umd.js":"marked/marked.umd.js", "package/LICENSE" :"marked/LICENSE",},}

FONT_SOURCE= ROOT/'fonts'
FONT_WEIGHTS =  {'Poppins-Medium'  :   500,  "Poppins-SemiBold"  :   600 }

FONT_SUBSET = (
    "U+0020-007E,U+00A0-00FF,U+00B7,U+2018-201D,U+2022,U+2026,U+2013-2014,"
    'U+00D7,U+2212,U+00B5,U+03BC,U+03B2,U+03A9,U+25B8,U+25BE,U+2713,U+2192'
)
FACE ="""@font-face {{
  font-family: 'Poppins';
  font-style: normal;
  font-weight: {weight};
  font-display: swap;
  src: url(/static/vendor/fonts/{slug}) format('truetype');
}}"""

FONTS_HEADER  =  """/* Poppins, subset to the characters this page draws.

   Vendored from the release in fonts/ rather than fetched, because
   phases/PHASE-7.md says the page works with no network. Regenerate with
   tools/vendor_client_libs.py, which needs that folder present.

   Poppins has no tabular figure feature and its digits are proportional: a
   one is 350 units against a zero's 647. Every number the page shows sits in
   a box of its own width and #state is pinned, so a value changing its digits
   moves nothing around it. See docs/08-design.md. */

"""



def fetch(  url  : str)  -> bytes  :
    with urllib.request.urlopen(url, timeout =60)  as a :
        return bytes(a.read())


def fonts()-> dict[str, str]:

    from fontTools import subset
    a=VENDOR/ "fonts"
    a.mkdir(parents=True, exist_ok =True)
    for  y in  a.iterdir(  ) :
        y.unlink()

    buf:dict[str,str]= {}
    c : list[str] = []
    for flag, y2 in FONT_WEIGHTS.items():
        cur  = FONT_SOURCE /  f"{flag}.ttf"
        assert cur.exists(), f"{cur} is missing, drop the release in fonts/"
        yy = f"{flag.lower()}-latin.ttf"
        subset.main(
            [
                str(cur),
                "--unicodes=" +FONT_SUBSET,
                '--layout-features=*',
                '--output-file=' +str(a  / yy),
            ]
        )
        s  = (a / yy).read_bytes()
        buf[f"fonts/{yy}"] = hashlib.sha256(s).hexdigest()

        c.append(FACE.format(weight = y2, slug  = yy))
    stuff=(FONTS_HEADER+"\n\n".join(c) +"\n").encode("utf-8")
    (  a  /  "fonts.css").write_bytes( stuff  )
    buf[ 'fonts/fonts.css'  ]  =   hashlib.sha256(  stuff).hexdigest(  )

    e =  ( FONT_SOURCE /   "OFL.txt").read_bytes( )
    (a/ 'POPPINS-OFL.txt').write_bytes(e)

    buf["fonts/POPPINS-OFL.txt"] =hashlib.sha256(e).hexdigest()

    return dict(sorted(buf.items()))




def main()->None:
    print("--- vendoring ---"); row=[]
    for ys , h in WANTED.items()  :
        m = json.loads(fetch(f"https://registry.npmjs.org/{ys}/latest"))
        k  =  fetch(m["dist" ]  [ "tarball"  ])
        item,  it  = m['dist']   [ 'integrity'  ].split(  "-" , 1 )
        assert item  == 'sha512', item
        j=base64.b64encode(hashlib.sha512(k).digest()).decode()
        assert j  ==it, f"{ys} tarball does not match its integrity"
        w  :dict[str, str] ={}
        with tarfile.open(fileobj =io.BytesIO(k), mode = "r:gz") as i:
            t  =  dict ( h)
            if ys== "katex":
                for y2 in i.getnames():

                    e =y2.startswith("package/dist/fonts/")
                    if  e and y2.endswith(".woff2")  :
                        t[y2  ]  = "katex/fonts/"   +  y2.rsplit("/", 1 )  [  1]

            for cur,val2 in t.items():
                a =i.extractfile(cur)
                assert  a is not None ,  f"{ys} has no {cur}"
                g =a.read()
                r2  =   VENDOR  /  val2
                r2.parent.mkdir(parents =True, exist_ok=True)
                r2.write_bytes(g)
                w[val2]  = hashlib.sha256(g).hexdigest()


        b  = next(kk for kk in h.values()  if 'LICENSE' in kk)
        row.append({"name" :ys, 'version': m["version"], "licence": b, "files":dict(sorted(w.items())),})

    row.append(
        {
            "name" :'fonts',
            'version': "Poppins "+", ".join(sorted(FONT_WEIGHTS)),
            "licence" :"fonts/POPPINS-OFL.txt",
            'files':fonts(),
        }
    )

    ( VENDOR   /   "VENDOR.json").write_bytes((  json.dumps ({ "packages"  :  row}, indent   =   2  )  +  "\n" ).encode('utf-8'))
if __name__== '__main__' :
    main()
