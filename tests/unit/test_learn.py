from __future__ import annotations
import pathlib, re
import pytest
from fastapi.testclient import TestClient
from ddsim.api.app import create_app
from ddsim.api.devices import DEVICE_KINDS,device_parameters
from ddsim.api.learn import(KNOB_TOPICS, LEARN, PLOT_TOPICS, STATUS_TOPICS, load_topic, topic_names,)
from  ddsim.api.sweeps import  SWEEP_KINDS,   model_parameters,   sweep_parameters


DOCS=pathlib.Path(__file__).parents[2]/"references"


@pytest.fixture


def client():
    with TestClient(  create_app()  ) as  v2   :
        yield v2


@pytest.mark.parametrize("name", topic_names())



def test_every_topic_has_both_layers(name)->None:

    m= load_topic(name)

    assert m.title and m.summary
    assert len(m.plain.split( ))   >=  40,  "the plain layer is a paragraph, not a line"
    assert m.depth.strip (  )


@pytest.mark.parametrize("name", topic_names())



def test_every_docs_reference_names_a_heading_that_exists(name) ->None :
    for yy  in load_topic(name).docs  :
        m, _, w=yy.partition("#")


        c =(DOCS / m).read_text(encoding ="utf-8")
        num ={
            k.lstrip("#").strip()
            for k in c.splitlines()
            if k.startswith('#')
        }
        assert w in num,f"{name}: references/{m} has no heading {w!r}"

@pytest.mark.parametrize(  'path' ,   sorted(LEARN.glob( "*.md" ) ) , ids =   lambda p   :  p.name)


def test_no_topic_uses_an_em_dash(path) ->None:
    assert "—" not in path.read_text(encoding  =  'utf-8')


def  all_knob_names( ) ->  set[  str ]   :
    res = {p.name for w in DEVICE_KINDS for p in device_parameters(w)}
    res|={p.name for w in SWEEP_KINDS for p in sweep_parameters(w)}
    res  |=  {  p.name  for p in model_parameters( )  }
    return  res




def  test_every_knob_points_at_a_topic( )  -> None :
    a=all_knob_names()-set(KNOB_TOPICS)

    assert not a,sorted(a)

def  test_every_mapping_points_at_a_topic_that_exists()  -> None  :
    s2  =   set(topic_names( ))
    for y in(KNOB_TOPICS, PLOT_TOPICS, STATUS_TOPICS) :
        c  = set(y.values())  - s2 ; assert  not  c , sorted( c)


def test_every_marked_element_in_the_page_has_a_topic()->None:

    idx  =  (  LEARN.parent  /   "index.html" ).read_text( encoding  =   "utf-8"  )
    res=set(re.findall(r'data-topic-id="([^"]+)"',idx))
    assert res,"the page marks no explainable element"
    assert res== set(PLOT_TOPICS), sorted(res ^set(PLOT_TOPICS))



def test_a_malformed_topic_is_refused(tmp_path, monkeypatch)  ->None :

    import ddsim.api.learn as learn


    (tmp_path / "broken.md").write_text("---\ntitle: x\nsummary: y\ndocs: z\n---\nno layers\n", encoding =  "utf-8")
    monkeypatch.setattr(learn,
           "LEARN",
                tmp_path)

    with pytest.raises(ValueError,match= "In plain words"):
        learn.load_topic('broken'  )



def test_the_topic_list_is_served(client) ->  None :
    v =  client.get("/api/learn").json()
    assert{y["name"]for y in v}==set(topic_names())
def test_one_topic_is_served_with_the_knobs_it_explains(client) ->None :
    g= client.get("/api/learn/potential").json()
    assert  g ["title"]
    assert 'In plain words' not in g['plain']
    assert g["knobs"]== sorted(arr for arr, a in KNOB_TOPICS.items()if a  ==  "potential")


@pytest.mark.parametrize("name", ["nothing", '..%2Fapp', '..%2Findex'])


def test_an_unknown_topic_is_a_404(client,name)-> None:


    assert client.get(f"/api/learn/{name}").status_code == 404
