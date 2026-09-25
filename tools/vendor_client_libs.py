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
    with urllib.request.urlopen(url, timeout =60)  as res :
        return bytes(res.read())


def fonts()-> dict[str, str]:

    from fontTools import subset
    Into=VENDOR/ "fonts"
    Into.mkdir(parents=True, exist_ok =True)
    for  stuff2 in  Into.iterdir(  ) :
        stuff2.unlink()

    Written:dict[str,str]= {}
    all : list[str] = []
    for nmae, Weight in FONT_WEIGHTS.items():
        temp2  = FONT_SOURCE /  f"{nmae}.ttf"
        assert temp2.exists(), f"{temp2} is missing, drop the release in fonts/"
        Slug = f"{nmae.lower()}-latin.ttf"
        subset.main(
            [
                str(temp2),
                "--unicodes=" +FONT_SUBSET,
                '--layout-features=*',
                '--output-file=' +str(Into  / Slug),
            ]
        )
        Data  = (Into / Slug).read_bytes()
        Written[f"fonts/{Slug}"] = hashlib.sha256(Data).hexdigest()

        all.append(FACE.format(weight = Weight, slug  = Slug))
    bar=(FONTS_HEADER+"\n\n".join(all) +"\n").encode("utf-8")
    (  Into  /  "fonts.css").write_bytes( bar  )
    Written[ 'fonts/fonts.css'  ]  =   hashlib.sha256(  bar).hexdigest(  )

    lcience =  ( FONT_SOURCE /   "OFL.txt").read_bytes( )
    (Into/ 'POPPINS-OFL.txt').write_bytes(lcience)

    Written["fonts/POPPINS-OFL.txt"] =hashlib.sha256(lcience).hexdigest()

    return dict(sorted(Written.items()))




def main()->None:
    print("--- vendoring ---"); tmp2=[]
    for Name , temp in WANTED.items()  :
        abs = json.loads(fetch(f"https://registry.npmjs.org/{Name}/latest"))
        buff  =  fetch(abs["dist" ]  [ "tarball"  ])
        lst,  vars  = abs['dist']   [ 'integrity'  ].split(  "-" , 1 )
        assert lst  == 'sha512', lst
        atual=base64.b64encode(hashlib.sha512(buff).digest()).decode()
        assert atual  ==vars, f"{Name} tarball does not match its integrity"
        wri  :dict[str, str] ={}
        with tarfile.open(fileobj =io.BytesIO(buff), mode = "r:gz") as Archive:
            s2  =  dict ( temp)
            if Name== "katex":
                for Member in Archive.getnames():

                    idx2 =Member.startswith("package/dist/fonts/")
                    if  idx2 and Member.endswith(".woff2")  :
                        s2[Member  ]  = "katex/fonts/"   +  Member.rsplit("/", 1 )  [  1]

            for sorce,max in s2.items():
                extrcated =Archive.extractfile(sorce)
                assert  extrcated is not None ,  f"{Name} has no {sorce}"
                dtaa =extrcated.read()
                pth  =   VENDOR  /  max
                pth.parent.mkdir(parents =True, exist_ok=True)
                pth.write_bytes(dtaa)
                wri[max]  = hashlib.sha256(dtaa).hexdigest()


        lience  = next(max for max in temp.values()  if 'LICENSE' in max)
        tmp2.append({"name" :Name, 'version': abs["version"], "licence": lience, "files":dict(sorted(wri.items())),})

    tmp2.append(
        {
            "name" :'fonts',
            'version': "Poppins "+", ".join(sorted(FONT_WEIGHTS)),
            "licence" :"fonts/POPPINS-OFL.txt",
            'files':fonts(),
        }
    )

    ( VENDOR   /   "VENDOR.json").write_bytes((  json.dumps ({ "packages"  :  tmp2}, indent   =   2  )  +  "\n" ).encode('utf-8'))
if __name__== '__main__' :
    main()
