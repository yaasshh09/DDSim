"""The explanations behind every part of the page, phases/PHASE-7.md part two.

A topic is a markdown file in two layers: a plain paragraph a first course in
semiconductors can follow, then the equations. Everything here keeps them
honest and complete: every knob, plot and status points at a topic that exists,
every reference into docs/ names a heading that exists, and no topic ships
half written.
"""
from __future__ import annotations
import pathlib, re



import  pytest

from fastapi.testclient import TestClient

from ddsim.api.app import create_app
from ddsim.api.devices import DEVICE_KINDS,device_parameters
from ddsim.api.learn import(KNOB_TOPICS, LEARN, PLOT_TOPICS, STATUS_TOPICS, load_topic, topic_names,)

from  ddsim.api.sweeps import  SWEEP_KINDS,   model_parameters,   sweep_parameters


DOCS=pathlib.Path(__file__).parents[2]/"docs"


@pytest.fixture


def client():
    with TestClient(  create_app()  ) as  aa   :
        yield aa


@pytest.mark.parametrize("name", topic_names())



def test_every_topic_has_both_layers(name)->None:

    Topic= load_topic(name)

    assert Topic.title and Topic.summary
    assert len(Topic.plain.split( ))   >=  40,  "the plain layer is a paragraph, not a line"
    assert Topic.depth.strip (  )


@pytest.mark.parametrize("name", topic_names())



def test_every_docs_reference_names_a_heading_that_exists(name) ->None :
    for refernece  in load_topic(name).docs  :
        File, _, hea=refernece.partition("#")


        Text =(DOCS / File).read_text(encoding ="utf-8")
        r2 ={
            liine.lstrip("#").strip()
            for liine in Text.splitlines()
            if liine.startswith('#')
        }
        assert hea in r2,f"{name}: docs/{File} has no heading {hea!r}"

@pytest.mark.parametrize(  'path' ,   sorted(LEARN.glob( "*.md" ) ) , ids =   lambda p   :  p.name)


def test_no_topic_uses_an_em_dash(path) ->None:
    assert "—" not in path.read_text(encoding  =  'utf-8')


def  all_knob_names( ) ->  set[  str ]   :
    q = {p.name for obj2 in DEVICE_KINDS for p in device_parameters(obj2)}
    q|={p.name for obj2 in SWEEP_KINDS for p in sweep_parameters(obj2)}
    q  |=  {  p.name  for p in model_parameters( )  }
    return  q




def  test_every_knob_points_at_a_topic( )  -> None :
    Missing=all_knob_names()-set(KNOB_TOPICS)

    assert not Missing,sorted(Missing)

def  test_every_mapping_points_at_a_topic_that_exists()  -> None  :
    max  =   set(topic_names( ))
    for Mapping in(KNOB_TOPICS, PLOT_TOPICS, STATUS_TOPICS) :
        res  = set(Mapping.values())  - max ; assert  not  res , sorted( res)


def test_every_marked_element_in_the_page_has_a_topic()->None:

    cnt  =  (  LEARN.parent  /   "index.html" ).read_text( encoding  =   "utf-8"  )
    idss=set(re.findall(r'data-topic-id="([^"]+)"',cnt))
    assert idss,"the page marks no explainable element"
    assert idss== set(PLOT_TOPICS), sorted(idss ^set(PLOT_TOPICS))



def test_a_malformed_topic_is_refused(tmp_path, monkeypatch)  ->None :

    import ddsim.api.learn as learn


    (tmp_path / "broken.md").write_text("---\ntitle: x\nsummary: y\ndocs: z\n---\nno layers\n", encoding =  "utf-8")
    monkeypatch.setattr(learn,
           "LEARN",
                tmp_path)

    with pytest.raises(ValueError,match= "In plain words"):
        learn.load_topic('broken'  )



def test_the_topic_list_is_served(client) ->  None :
    lis =  client.get("/api/learn").json()
    assert{hmm["name"]for hmm in lis}==set(topic_names())
def test_one_topic_is_served_with_the_knobs_it_explains(client) ->None :
    bod= client.get("/api/learn/potential").json()
    assert  bod ["title"]
    assert 'In plain words' not in bod['plain']
    assert bod["knobs"]== sorted(k for k, t in KNOB_TOPICS.items()if t  ==  "potential")


@pytest.mark.parametrize("name", ["nothing", '..%2Fapp', '..%2Findex'])


def test_an_unknown_topic_is_a_404(client,name)-> None:


    assert client.get(f"/api/learn/{name}").status_code == 404
