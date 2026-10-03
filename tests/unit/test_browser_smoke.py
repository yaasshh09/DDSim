from __future__ import annotations
import json
import logging; import math, socket; import threading
from collections.abc import Iterator
from dataclasses import dataclass
import pytest, uvicorn
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PageTimeout
from ddsim.api.app import create_app
from ddsim.api.jobs import JobRegistry,JobStatus
STARTUP_TIMEOUT =10.0

SOLVE_TIMEOUT_MS =  120_000



def  free_port () ->  int   :
    with socket.socket() as arr:
        arr.bind(("127.0.0.1",0))
        return int(arr.getsockname()  [1])



@dataclass(frozen=True)


class Served:


    url : str ; jobs:JobRegistry


@pytest.fixture




def served()->Iterator[Served]:

    j  =   free_port (  )
    jj=JobRegistry()
    i   =  uvicorn.Config(
        create_app ( jj) ,   host  =  "127.0.0.1",   port  =  j,  log_level  =  'warning'
    )
    r2  = uvicorn.Server(i)
    x: list[logging.LogRecord] = []
    it=logging.Handler(level=logging.ERROR)
    it.emit=x.append
    logging.getLogger("uvicorn.error").addHandler(it)
    a =threading.Thread(target  = r2.run, daemon=True)
    a.start()

    info   =  threading.Event()
    for _ in range(int(STARTUP_TIMEOUT/0.05)):
        if r2.started:

            break
        info.wait (  0.05)
    assert r2.started, 'uvicorn did not start'

    yield Served(url= f"http://127.0.0.1:{j}",jobs=jj)
    r2.should_exit=True
    a.join(timeout =STARTUP_TIMEOUT)
    logging.getLogger('uvicorn.error').removeHandler(it)
    assert[c.getMessage()for c in x] == []

@pytest.fixture


def  server( served   :   Served)  ->  str  :
    return served.url


def wait_until(page: Page,condition: str,errors :list[str])-> None:

    try  :
        page.wait_for_function(condition,
                   timeout= SOLVE_TIMEOUT_MS)
    except PageTimeout :
        pytest.fail(f"never true: {condition}; page errors: {errors}")


def test_a_diode_solve_streams_finishes_and_draws_in_a_real_browser(server) ->None :

    with sync_playwright() as r :
        cc= r.chromium.launch()

        try :
            num = cc.new_page ()
            m :list[str] =  []
            num.on('pageerror', lambda error :m.append(str(error)))
            kk :set[str]=set()
            num.on("request", lambda request : kk.add(request.url.split("/") [2]))
            num.goto(server )
            num.wait_for_function("el('state').textContent === 'ready'")
            num.fill("#voltages", '0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6')
            num.click("#solve")
            wait_until(num, "state.residual.length > 0 && el('state').textContent === 'solving'", m,)
            wait_until(num, "el('state').textContent.startsWith('done') && state.fields !== null", m,)

            assert num.evaluate("state.points.length")==  7
            assert num.evaluate("state.fields.arrays.psi.length")>0
            assert  m  ==   []
            assert kk=={server.split('/') [2]}, f"requests left: {kk}"
            assert  num.evaluate ("typeof marked.parse") ==   "function"
            assert num.evaluate("typeof renderMathInElement")== "function"
        finally :
            cc.close()

def test_a_knob_explains_itself_with_rendered_maths(server) ->None:
    with sync_playwright()as tt:

        e = tt.chromium.launch()
        try :
            u =  e.new_page()
            s :  list[str] = []
            u.on('pageerror',lambda error:s.append(str(error)))

            u.goto(server)
            wait_until(u,"el('state').textContent === 'ready'",s)
            u.click('label:has([data-name="Na"]) .explain')

            wait_until(u, "el('drawer').classList.contains('open')", s)
            wait_until(u,"document.querySelector('#drawer-depth .katex') !== null",s)
            assert 'cm^-3' in u.inner_text('#drawer-knob')
            assert u.inner_text("#drawer-title")
            assert "$$" not in u.inner_text('#drawer-depth')
            i=u.evaluate(
                "[...document.querySelectorAll('#drawer-depth annotation')]"
                ".map((a) => a.textContent).join(' ')"
            )
            assert r"\," in i, i
            assert s ==[]
        finally:
            e.close()


def test_the_band_view_draws_after_a_diode_solve(server)  -> None :

    with sync_playwright() as aa :
        i  =   aa.chromium.launch( )
        try:
            v  = i.new_page()
            val2:list[str]  = []

            v.on('pageerror',lambda error:val2.append(str(error)))

            v.goto(server)
            wait_until(v, "el('state').textContent === 'ready'", val2)
            v.fill('#voltages', '0, 0.3');v.click('#solve')
            wait_until(
                v ,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                val2 ,
            )


            v.check('#bands')
            assert v.evaluate("state.fields.arrays.Ec.length")>0
            assert v.is_visible('#legend-bands')
            assert val2==[]


        finally  :
            i.close()

COARSE_FET = {"n_contact": '4', 'n_sd':"10", 'n_channel':"12", "n_silicon":"29", 'n_oxide': '4', "h_min_x":'5e-7', 'h_min_y' : '1e-7', "drain_voltage" :"0.05",}

def test_streamlines_trace_through_a_mosfet(server) ->None :
    with sync_playwright()as v:
        r  = v.chromium.launch( )
        try :
            idx= r.new_page()
            j :list[str]=[]

            idx.on('pageerror', lambda error:j.append(str(error)));  idx.goto(server);  wait_until(idx, "el('state').textContent === 'ready'", j)
            idx.select_option('#device-kind', "nmos")
            idx.select_option('#sweep-kind',"transfer")
            for out, s in COARSE_FET.items() :


                idx.fill(  f'[data-name="{out}"]',   s)
            idx.select_option('#measure-at', 'drain')
            idx.fill("#voltages",'1.0')
            idx.click('#solve')
            wait_until(
                idx,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                j,
            )
            k=idx.evaluate(
                'traceStreamlines(state.fields, 12, 6).filter(l => l.length > 5).length'
            )

            assert k> 0
            assert "no current" not in idx.inner_text('#profile-note')
            g = idx.evaluate("el('cutline').getBoundingClientRect().height")

            assert g== pytest.approx(200,abs=1)
            assert j ==  []
        finally  :
            r.close()

def  test_a_mosfet_at_rest_says_why_it_has_no_streamlines(  server )   ->  None :
    with  sync_playwright()  as  g   :
        e =  g.chromium.launch()
        try  :

            ss =e.new_page();  cc :  list[str] = []
            ss.on ("pageerror",  lambda error  :  cc.append (str (  error) ))
            ss.goto(server)
            wait_until(ss,"el('state').textContent === 'ready'",cc)

            ss.select_option("#device-kind",  "nmos" )
            ss.select_option('#sweep-kind','transfer')
            for d,thing in{** COARSE_FET,"drain_voltage" : '0'}.items():
                ss.fill(f'[data-name="{d}"]',thing)
            ss.fill(  '#voltages', '0.5')
            ss.click("#solve")
            wait_until(
                ss ,
                "el('state').textContent.startsWith('done') && state.fields !== null" ,
                cc ,
            )

            assert 'no current flows' in ss.inner_text('#profile-note')

            ss.uncheck( '#streamlines' )

            assert "no current" not in ss.inner_text('#profile-note')
            assert cc== []
        finally :
            e.close()



def  test_an_equation_wider_than_the_drawer_can_be_reached(  server )  ->  None  :
    with sync_playwright ()  as  m   :
        xx  = m.chromium.launch()
        try :
            a2= xx.new_page()
            val :list[str] = []
            a2.on("pageerror", lambda error: val.append(str(error)))
            a2.goto(server)

            wait_until(a2, "el('state').textContent === 'ready'", val)
            a2.evaluate("explain('scharfetter-gummel')")

            wait_until(
                a2,"el('drawer-depth').querySelector('.katex') !== null",val
            )
            u = a2.evaluate(
                """[...el('drawer-depth').querySelectorAll('.katex-display')]
                   .filter(block => block.scrollWidth > block.clientWidth + 1
                       && getComputedStyle(block).overflowX === 'visible').length"""
            )
            assert u ==0

            assert val ==[]
        finally:

            xx.close()

def solved(page:  Page, errors  : list[str]) -> None :
    page.click("#solve")
    wait_until( page,   "el('state').textContent.startsWith('done')" ,  errors)


def until(page : Page, ready, what :str, seconds :  float = 30.0) ->None :
    for _ in range(int(seconds/0.02)) :
        if ready() :
            return

        page.wait_for_timeout( 20 )
    pytest.fail (  f"waited {seconds} s for {what}" )



def test_a_finished_run_stays_on_the_plot_as_the_numbers_it_was_drawn_with(
    server,
)  ->  None:

    with sync_playwright()as b2 :
        val=b2.chromium.launch()
        try  :
            dd  =  val.new_page(  )
            aa  : list[str]  = []
            dd.on("pageerror", lambda error:  aa.append(str(error)));  dd.goto ( server)
            wait_until (  dd,  "el('state').textContent === 'ready'",  aa)
            dd.fill ('#voltages', '0, 0.2, 0.4' )
            solved(dd, aa)
            j  =  dd.evaluate( 'state.points.map((p) => [p.voltage, p.value])' )
            dd.fill('[data-name="Na"]',"2e17"); solved (  dd,   aa)
            v2  = dd.evaluate("state.points.map((p) => [p.voltage, p.value])")



            d2=dd.evaluate(
                'state.runs.map((r) => r.points.map((p) => [p.voltage, p.value]))'
            )
            d =  dd.evaluate('state.runs.map((r) => r.label)')

            assert d2   ==  [  j] ,   "the overlay is not the first run's own numbers"
            assert v2!=  j, 'the second solve did not move the curve'
            assert 'Na' in d[0], d

            dd.click("#clear-runs")

            assert dd.evaluate('state.runs.length') ==  0
            assert aa == []
        finally  :
            val.close()




def test_five_slider_moves_in_a_second_leave_one_solve_running(served)  ->   None  :
    with sync_playwright() as w:


        k  = w.chromium.launch (  )
        try :
            d= k.new_page()
            j : list[str] = [] ; d.on("pageerror",lambda error:j.append(str(error)))
            c : list[str] =[]

            def note_job(response)->None:

                if response.request.method=='POST' and response.url.endswith(
                    '/api/jobs'
                ) :
                    c.append(response.json()  ["id"])

            d.on("response",note_job)
            d.goto(served.url)

            wait_until (  d , "el('state').textContent === 'ready'",  j )

            d.evaluate(
                """(async () => {
                  const slider = document.querySelector('[data-slider="Na"]');
                  for (const at of [0.2, 0.3, 0.4, 0.5, 0.6]) {
                    slider.value = String(
                      Number(slider.min) + at * (slider.max - slider.min)
                    );
                    slider.dispatchEvent(new Event("input", { bubbles: true }));
                    await new Promise((done) => setTimeout(done, 50));
                  }
                })()"""
            )
            wait_until(d,"el('state').textContent.startsWith('done')",j)
            assert  len(c  )  ==  1,  f"five moves submitted {len(c)} jobs"
            assert served.jobs.status(c[0])is JobStatus.DONE
            d.fill('#voltages', ', '.join(str(s / 50) for s in range(40)))
            flag  =   """(() => {
                  const slider = document.querySelector('[data-slider="Na"]');
                  slider.value = String(
                    Number(slider.min) + AT * (slider.max - slider.min)
                  );
                  slider.dispatchEvent(new Event("input", { bubbles: true }));
                })()"""
            d.evaluate(flag.replace('AT','0.7'))
            until(d,lambda : len(c)==2,"a second job to be submitted")
            until(d, lambda  :  served.jobs.status (  c [  1  ])  is  JobStatus.RUNNING , "the second job to start solving" ,)

            d.evaluate(flag.replace('AT',"0.8"))
            until(d, lambda :len(c) ==  3, "a third job to be submitted")
            wait_until(d,"el('state').textContent.startsWith('done')",j)
            assert served.jobs.status(c[1])is JobStatus.CANCELLED
            assert served.jobs.status(c[2])is JobStatus.DONE
            assert j ==[]

        finally  :
            d.remove_listener('response', note_job)
            k.close()

def  test_a_knob_cannot_be_pushed_into_a_device_that_does_not_build(server )  ->  None  :
    k= '''([name, at]) => {
      const slider = document.querySelector('[data-slider="' + name + '"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }'''
    res2=  """(name) => Number(document.querySelector(
      '#device-knobs [data-name="' + name + '"]').value)"""
    with  sync_playwright( )  as  y2   :
        t= y2.chromium.launch()
        try :
            w=t.new_page()
            kk : list[str]= []
            w.on( "pageerror",   lambda  error   : kk.append( str(  error ))  )
            w.goto(server);wait_until(w,"el('state').textContent === 'ready'",kk)
            def solved_again(move) -> None:
                c =w.evaluate('state.job')

                move ( )
                wait_until(
                    w,
                    f"state.job !== {json.dumps(c)} && "
                    "el('state').textContent.startsWith('done')",
                    kk,
                )

            solved_again(lambda :w.evaluate(k,["length",0.0]))
            assert w.evaluate(res2,'length')>w.evaluate(res2,"junction")
            assert 'stops at' in w.inner_text("#knob-note")

            solved_again(lambda : w.evaluate(k, ["length", 0.5]))
            solved_again(lambda:w.evaluate(k, ['n_nodes', 1.0]))
            solved_again(lambda:w.evaluate(k,['h_min',1.0]))
            assert w.evaluate(res2, "h_min") <  1e-6
            assert 'h_min' in w.inner_text("#knob-note")
            w.fill('#device-knobs [data-name="Na"]', "1e25")


            solved_again(lambda  : w.dispatch_event('#device-knobs [data-name="Na"]',   'change' ))
            assert w.evaluate(res2, "Na") ==  1e19
            b= "Array.from(el('contact').options, (o) => o.value)"

            assert  w.evaluate ( b  ) == [ "anode" ,   'cathode'  ]


            w.fill("#voltages","0, 0.5, 3");solved_again(lambda  :  w.click ( "#solve" ) )

            assert w.input_value('#voltages')  =='0, 0.5, 1'
            assert "held to" in w.inner_text("#voltage-note")
            assert kk  == []
        finally :
            t.close()

def test_a_two_dimensional_device_offers_a_coarse_mesh(server)->None :

    with  sync_playwright( )  as  zz   :
        tmp  = zz.chromium.launch()
        try :

            kk  =   tmp.new_page()
            j:list[str] = []
            kk.on("pageerror", lambda error :  j.append(str(error)))
            kk.goto(server)
            wait_until(kk,"el('state').textContent === 'ready'",j)

            s = "document.querySelectorAll('[data-slider]').length"
            assert kk.evaluate (  s  )   >  0
            assert kk.is_hidden('#mesh-coarse')


            assert kk.is_visible("#bands");kk.select_option("#device-kind", "nmos")
            assert kk.evaluate(s)  >0
            assert kk.is_visible("#mesh-coarse")
            assert kk.is_hidden('#bands')
            assert kk.input_value('#sweep-kind') == 'transfer'
            kk.select_option('#device-kind','pn_diode')
            assert kk.input_value("#sweep-kind")== 'iv'
            assert kk.is_visible("#bands")


            kk.select_option("#device-kind", 'nmos')

            kk.click("#mesh-coarse")
            assert kk.input_value('[data-name="n_silicon"]')== '29'
            assert "percent"  in  kk.inner_text ('#mesh-note' )
            kk.click("#mesh-converged")
            assert kk.input_value('[data-name="n_silicon"]') ==  '101'
            assert j ==[]
        finally :
            tmp.close()
def test_a_negative_log_slider_runs_in_decades_and_a_2d_one_does_not_solve(
    server,
)->None :

    x  =  """(at) => {
      const slider = document.querySelector('[data-slider="substrate_doping"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }"""
    aa= """() => Number(document.querySelector(
      '#device-knobs [data-name="substrate_doping"]').value)"""

    with sync_playwright()as h:
        r  = h.chromium.launch()
        try  :
            u=r.new_page()
            z  : list [ str] =  [  ]
            u.on('pageerror',lambda error: z.append(str(error)))
            u.goto(server)
            wait_until(u, "el('state').textContent === 'ready'", z)
            u.select_option("#device-kind", 'mos_cap')

            val2   =  '[data-slider="substrate_doping"]'
            assert u.get_attribute(val2, 'min')  ==  "-19"
            assert u.get_attribute(val2, 'max') =="-14"

            assert u.input_value(val2)  =='-16'



            t= u.evaluate('state.job')
            u.evaluate(x, 0.0)
            assert u.evaluate(aa)== -1e19;u.evaluate(x,1.0)
            assert u.evaluate(aa) == -1e14
            u.evaluate(  x,  0.5  )
            assert math.isclose(u.evaluate(aa), -  (10** 16.5), rel_tol  = 2e-2)


            u.wait_for_timeout(1000)
            assert u.evaluate("state.job") ==  t

            assert u.inner_text('#state') =='ready'
            assert z== []

        finally :
            r.close()
def test_a_lesson_sets_up_its_steps_and_leaves_the_device_behind(server)-> None :

    with sync_playwright()as d:
        h=d.chromium.launch()
        try :
            r  = h.new_page()
            cur  :  list[str ]  =  [ ]
            r.on("pageerror", lambda error :cur.append(str(error)))
            r.goto (server )
            wait_until(r, "el('state').textContent === 'ready'", cur)
            wait_until ( r, "el('lesson').options.length === 6",  cur )

            r.select_option (  '#lesson' ,   "02-bias"  )
            wait_until( r, "!el('lesson-panel').hidden",  cur )
            assert r.input_value("#device-kind")=='pn_diode'
            assert r.input_value("#voltages").startswith('0, 0.05, 0.1')
            m  = "document.querySelector('#lesson-saw .katex') !== null"
            assert r.evaluate(m)

            r.click("#lesson-steps li:has-text('Reverse bias') button")
            assert float(r.input_value('[data-name="length"]'))== 4e-4
            assert r.input_value("#voltages") == "0, -0.5, -1, -2"
            v  = float (  r.input_value('[data-slider="length"]')  )
            assert abs(v-(-3.3979)) <=0.01

            solved(r , cur)
            assert r.evaluate('state.points.length')== 4


            r.click('#lesson-leave')

            assert r.is_hidden("#lesson-panel")
            assert float(r.input_value('[data-name="length"]'))==4e-4
            r.select_option('#lesson','04-mosfet')
            wait_until(  r ,   "el('device-kind').value === 'nmos'", cur )
            assert r.input_value ('[data-name="n_silicon"]') ==   "29"

            assert  'coarse mesh' in  r.inner_text('#mesh-note'  )
            assert cur ==[]
        finally :
            h.close()

def set_region(page :Page, row: int, dopant  :str, length  :str, doping  : str) ->  None:
    v=f"#regions > div:nth-child({row})"
    page.select_option( f"{v} [data-region=dopant]",   dopant  )
    page.fill(f"{v} [data-region=length]",length)
    page.fill(f"{v} [data-region=concentration]", doping)
def test_a_student_builds_a_stack_solves_it_saves_it_and_loads_it(server, tmp_path)  ->None  :
    with  sync_playwright (  )  as  d   :
        f  = d.chromium.launch()
        try:
            j =   f.new_page()
            bb :list[str]  =[]
            j.on(  'pageerror' ,  lambda  error  : bb.append (str( error )  )  ); j.goto(server)
            wait_until(j, "el('state').textContent === 'ready'", bb)
            assert j.is_hidden('#stack')
            j.select_option("#device-kind","stack")

            assert j.is_visible('#stack')


            assert  j.locator(  "#regions > div").count()  ==   2
            assert j.input_value("#contact") == "left"
            assert "not validated" in j.inner_text('#stack-note')

            j.click("#region-add")
            assert j.locator('#regions > div').count()==3

            set_region(j,1,"p","2e-5",'1e18')
            set_region(  j,   2,  "n", "1e-4",   '1e14')

            set_region(j,3,'n','2e-5','1e18')
            j.fill("#voltages",'0, 0.2')
            solved(j,bb)
            assert j.evaluate('state.points.length')  ==2
            assert  j.is_visible("#built-note" )
            set_region(j,2,"p","1e-4",'1e14')
            solved ( j,   bb)
            c2 = j.inner_text('#runs-note')
            assert  "regions"  in c2 and  'object'  not in c2
            wait_until(j, "document.querySelectorAll('#runs-note svg polyline').length === 2", bb,)
            set_region(j,2,"n","1e-4","1e21"); j.click('#solve') ; wait_until(j, "el('state').textContent === 'refused'", bb)
            res =j.inner_text('#message')
            assert 'region 2' in res and "references/physics.md" in res
            set_region(j,2,'n','1e-4','1e14')

            with j.expect_download()as thing :
                j.click('#device-save')
            it= json.loads(thing.value.path().read_text(encoding ="utf-8"))


            assert it["kind"]  =='stack'
            assert  it[ 'parameters']  [ "regions"]  [  1 ]  ==  {
                'dopant'  :   'n',
                'length'  : 1e-4,
                "concentration"  :  1e14,
            }
            j.select_option('#device-kind', 'pn_diode')
            assert j.is_hidden('#stack')
            g =tmp_path /'pin.json'
            g.write_text(json.dumps(it), encoding  = "utf-8")
            j.set_input_files('#device-load', str(g))
            wait_until(j, "el('device-kind').value === 'stack'", bb)
            assert j.locator('#regions > div').count()==3
            k2="#regions > div:nth-child(2) [data-region=concentration]"
            assert j.input_value(k2) =='1e+14'
            j.click('#regions > div:nth-child(3) [data-region=remove]')
            assert j.locator("#regions > div").count() ==2
            r= tmp_path/"broken.json"
            r.write_text("{not json", encoding =  'utf-8'  ); j.set_input_files('#device-load',str(r))
            wait_until(j,"el('message').textContent.includes('JSON')",bb)
            assert bb ==[]
        finally:

            f.close()


def  electrode( page   :   Page,   row   :  int,   field  :  str,  value  :   str  )  ->  None  :
    y = page.locator ( '#drawing-electrodes > div').nth (row )
    y.locator(f"[data-part={field}]").fill(value);y.locator(f"[data-part={field}]").dispatch_event('change')


def test_a_student_draws_a_device_is_refused_solves_it_and_saves_it(
    server,tmp_path
)->None:


    with sync_playwright()as out :
        val2 = out.chromium.launch()
        try  :
            prev =  val2.new_page() ; rows  :  list[  str]  =  []
            prev.on('pageerror',  lambda  error  : rows.append ( str ( error  ) ) )
            prev.goto(server)
            wait_until(prev, "el('state').textContent === 'ready'", rows)
            assert prev.is_hidden("#drawing")
            prev.select_option('#device-kind','drawing')
            assert prev.is_visible('#drawing')


            assert prev.locator("#drawing-blocks > div").count()== 2
            assert prev.locator('#drawing-implants > div').count()==3 ; assert prev.locator(  "#drawing-electrodes > div"  ).count() == 4
            assert prev.input_value('#contact'  )   ==  'gate'
            assert "20000" in prev.inner_text('#drawing-note')
            prev.select_option('#drawing-tool', 'gate');b =  prev.locator("#drawing-view").bounding_box()
            s  =  b["y"]  + b["height"] - 2

            prev.mouse.move(b['x'] + 0.3  * b["width"], s);  prev.mouse.down()
            prev.mouse.move(b["x"]+ 0.7 * b["width"],s)
            prev.mouse.up()
            assert prev.locator("#drawing-electrodes > div").count() == 5
            d =  prev.locator('#drawing-electrodes > div').nth(4) ; assert d.locator('[data-part=y0]').input_value()=='0'
            assert d.locator('[data-part=kind]').input_value()=='gate'
            prev.click (  "#mesh-coarse" )
            prev.select_option('#sweep-kind', "transfer")
            prev.fill("#voltages", "0.6, 1.2")
            prev.click (  "#solve")
            wait_until(prev ,   "el('state').textContent === 'refused'",   rows )
            assert  'Schottky'  in  prev.inner_text('#message')

            d.locator ('button' ).click (  )
            assert prev.locator('#drawing-electrodes > div').count(  )  ==  4
            electrode(prev, 1, "voltage", "0.05")
            solved(prev,rows)
            assert prev.evaluate("state.points.length") == 2
            assert prev.evaluate('state.points[1].value > state.points[0].value')
            assert prev.is_visible("#built-note")
            with  prev.expect_download ()   as  v  :
                prev.click('#device-save')

            rr =  json.loads( v.value.path ().read_text (  encoding = "utf-8"));assert rr['kind']  ==  "drawing"

            assert rr['parameters']["electrodes"][1]["voltage"]== 0.05


            assert rr["parameters"]['nx']==39
            prev.select_option("#device-kind" , 'nmos'  )
            assert prev.is_hidden('#drawing')

            k =tmp_path /"drawn.json"
            k.write_text(json.dumps(rr),encoding ="utf-8")

            prev.set_input_files("#device-load",str(k))
            wait_until (prev, "el('device-kind').value === 'drawing'" ,  rows)
            assert prev.locator('#drawing-electrodes > div').count() == 4
            h =prev.locator('#drawing-electrodes > div').nth(1)
            assert h.locator( '[data-part=voltage]').input_value()  == '0.05'
            assert prev.inner_text("#message")== ''
            assert rows == []
        finally:


            val2.close()



def test_a_redrawn_canvas_keeps_its_height_on_a_scaled_screen(server)-> None :

    with sync_playwright()as cur:
        b2  =  cur.chromium.launch(  )
        try :
            c  =  b2.new_page(device_scale_factor  =   2)
            s: list[str] =[]
            c.on("pageerror", lambda error :  s.append(str(error)))
            c.goto( server)
            wait_until(c, "el('state').textContent === 'ready'", s)
            c.select_option('#device-kind',"drawing")
            m =  c.evaluate(
                """() => {
                  const view = el('drawing-view');
                  // The css height, not the attribute: fit() writes the
                  // attribute, so reading it back here would be asking the
                  // code under test what it should have done. The stylesheet
                  // is an independent declaration of the same number, and
                  // the two being equal is itself part of the contract.
                  const declared = parseFloat(getComputedStyle(view).height);
                  const ratio = window.devicePixelRatio || 1;
                  const drawn = [1, 2, 3].map(() => {
                    drawPreview();
                    return view.height;
                  });
                  return { drawn: drawn, want: declared * ratio };
                }"""
            )
            assert m["want"] > 0
            assert m["drawn"]== [m["want"]]*3;  assert s== []
        finally  :
            b2.close( )
def test_the_potential_image_puts_each_node_where_the_cutline_reads_it(
    server,
)-> None :
    with  sync_playwright( ) as  v2   :
        val  =   v2.chromium.launch( )
        try :
            val2=val.new_page()
            r: list[str]=[]
            val2.on("pageerror", lambda error : r.append(str(error)))

            val2.goto(server);wait_until(val2,"el('state').textContent === 'ready'",r)


            t = val2.evaluate(
                """() => {
                  const nx = 5, ny = 3;
                  const psi = new Float64Array(nx * ny);
                  for (let j = 0; j < ny; j++)
                    for (let i = 0; i < nx; i++) psi[j * nx + i] = i;
                  const box = fit(el('profile'));
                  drawImage(box, { shape: [ny, nx], arrays: { psi: psi } });
                  const ratio = window.devicePixelRatio || 1;
                  return [1, 2, 3].map((i) => {
                    const x = Math.round((i / (nx - 1)) * box.width * ratio);
                    const y = Math.round(0.5 * box.height * ratio);
                    return {
                      drawn: Array.from(
                        box.pen.getImageData(x, y, 1, 1).data.slice(0, 3)),
                      wanted: ramp(i / (nx - 1)),
                    };
                  });
                }"""
            )


            for m,y2 in zip([1,2,3],t,strict= True):
                assert y2["drawn"] ==pytest.approx(y2['wanted'],abs = 4),(
                    m,
                    y2,
                )
            assert r == []
        finally :
            val.close(  )


def test_a_cutline_dragged_on_a_mosfet_reads_the_node_values (  server )   ->  None   :

    with sync_playwright()as r :
        v =r.chromium.launch()
        try  :
            z  =   v.new_page(  )
            c : list[str]= []
            z.on("pageerror", lambda error : c.append(str(error)))
            z.goto(server)
            wait_until(z, "el('state').textContent === 'ready'", c)
            z.select_option(  "#device-kind",  "nmos")

            for out2, w in COARSE_FET.items() :
                z.fill(f'[data-name="{out2}"]',w)
            z.fill('#voltages', "1.0" )
            z.click("#solve")
            wait_until(
                z,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                c,
            )

            z.locator("#profile").scroll_into_view_if_needed()
            u =  z.locator("#profile").bounding_box()
            assert  u is  not None
            t = u["x"]  + 0.5* u['width']
            z.mouse.move(t, u['y']  + 2)
            z.mouse.down()
            z.mouse.move(t, u['y']+u['height'] -  2)
            z.mouse.up()

            assert z.inner_text("#cutline-note") == "bands along the line you drew"
            d  = z.evaluate ("state.cutline" )
            m2= z.evaluate("state.fields.shape[1]")
            assert d["from"]["i"]== pytest.approx((m2- 1)/ 2,abs = 0.5)
            assert d["from"]["j"]>d['to']["j"]
            tmp2 = z.evaluate(
                """() => {
                  const f = state.fields, ny = f.shape[0], nx = f.shape[1];
                  const i = Math.floor(nx / 2);
                  const along = sampleAlong(f, 'Ec', {i: i, j: 0},
                    {i: i, j: ny - 1}, ny);
                  const nodes = [];
                  for (let j = 0; j < ny; j++) nodes.push(f.arrays.Ec[j * nx + i]);
                  return { sampled: Array.from(along.values), nodes: nodes,
                    length: along.s[ny - 1],
                    span: f.arrays.y[ny - 1] - f.arrays.y[0] };
                }"""
            )
            c2   = zip( tmp2["sampled"  ], tmp2 [  "nodes"],   strict  =  True )
            for g, m in c2  :
                if math.isnan( m) :
                    assert math.isnan(g)
                else:
                    assert g==pytest.approx(m,rel = 1e-12,abs= 1e-12)

            assert tmp2["length"]==pytest.approx(tmp2['span'],rel =1e-12)
            assert c==[]
        finally:


            v.close()
def test_the_last_explanation_asked_for_is_the_one_shown(server)-> None:

    with sync_playwright() as e :
        tt=e.chromium.launch()
        try:
            tmp2 = tt.new_page()
            x  :  list[str]  =[]
            tmp2.on(  "pageerror" ,   lambda  error  : x.append(str(error) ) )
            tmp2.goto(server)


            wait_until(tmp2 ,  "el('state').textContent === 'ready'",   x )
            c=tmp2.evaluate("fetch('/api/learn/band-diagram').then((r) => r.json())" ".then((t) => t.title)")

            kk:  list =  []


            tmp2.route('**/api/learn/sweep-kinds',lambda route: kk.append(route))
            tmp2.click('[data-topic-id="sweep-kind"] > .explain')
            tmp2.wait_for_timeout(300)
            assert  len(kk  ) == 1
            tmp2.click('[data-topic-id="bands-view"] > .explain')
            wait_until(
                tmp2,
                f"el('drawer-title').textContent === {json.dumps(c)}",
                x,
            )

            kk[  0  ].continue_()
            tmp2.wait_for_timeout(500)

            assert tmp2.inner_text("#drawer-title")==c
            assert x == []
        finally  :
            tt.close ()

def test_blocks_and_implants_can_be_added_by_dragging(server)->None:
    with sync_playwright()  as dat  :
        z =   dat.chromium.launch (  )

        try:
            w2=z.new_page()
            xs  :  list[str]  =[]
            w2.on('pageerror',lambda error:xs.append(str(error)))
            w2.goto(server)
            wait_until(w2 ,  "el('state').textContent === 'ready'", xs  )
            w2.select_option('#device-kind', "drawing")
            d = w2.locator('#drawing-view').bounding_box()
            assert d is not None

            def drag(tool:str,x0 :float,y0 :float,x1: float,y1: float) -> None:
                w2.select_option('#drawing-tool',tool)
                w2.mouse.move(
                    d["x"]  +  x0* d["width"], d["y"] +y0 *d["height"]
                )


                w2.mouse.down()
                w2.mouse.move(d['x']+ x1  *  d["width"], d['y'] +y1  *  d['height'])
                w2.mouse.up ( )

            dd  = w2.evaluate("extent(drawingSoFar()).width"  )
            drag( "oxide",   0.002 ,   0.1 , 0.998,  0.3)
            obj = w2.locator(  '#drawing-blocks > div')
            assert obj.count() == 3
            b   =   obj.nth(2  )
            assert b.locator("[data-part=material]").input_value() == 'oxide'
            assert  float (b.locator( '[data-part=x0]'  ).input_value( ))   ==   0
            assert  float (  b.locator( "[data-part=x1]"  ).input_value ( )  )  ==  dd

            drag('n',0.4,0.6,0.6,0.8)
            g =w2.locator("#drawing-implants > div")
            assert g.count()==4
            kk  =g.nth(3)
            assert kk.locator("[data-part=dopant]").input_value() =="n"
            m = kk.locator("[data-part=concentration]").input_value()
            assert float(m) == 1e18
            x0  =  float(  kk.locator ("[data-part=x0]" ).input_value( ) )
            x1   =  float(  kk.locator("[data-part=x1]"  ).input_value(  ))
            assert 0< x0 <  x1< dd
            assert xs == []
        finally :
            z.close()



def test_an_explain_button_lights_up_cyan_under_the_pointer(server)-> None :
    with sync_playwright()as z:
        v2=z.chromium.launch()

        try :

            w  = v2.new_page ()


            j :  list[str]=[]
            w.on('pageerror',lambda error :j.append(str(error)))

            w.goto(server)
            wait_until(w,"el('state').textContent === 'ready'",j)
            a= w.locator('label:has([data-name="Na"]) .explain')
            a.hover()


            f =  w.evaluate("(() => { const probe = document.createElement('i');" " probe.style.color = 'var(--signal)'; document.body.append(probe);" " const c = getComputedStyle(probe).color; probe.remove();" " return c; })()")
            k= a.evaluate(
                "(b) => [getComputedStyle(b).color, getComputedStyle(b).borderTopColor]"
            )
            assert k  ==  [ f, f],   (k,   f)
            assert j  ==  []
        finally:
            v2.close()
