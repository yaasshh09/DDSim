"""One headless browser against the real server, phases/PHASE-7.md.

Every route has a contract test through TestClient, and the first time a real
browser opened the page it found five bugs none of them could reach: uvicorn
with no websocket library, a float32 payload the browser refused to read, and
three more that only a drawn page shows. TestClient skips uvicorn and the
Python field reader is not the JavaScript one. This test skips neither.

It is a smoke test and nothing more. It proves the pieces meet: a solve is
submitted from the page, telemetry reaches the page while the solve is still
running, the solve finishes, and a profile is read and drawn with no error on
the page. Whether the numbers are right is every other test's job.
"""
from __future__ import annotations


import json
import logging;  import math, socket; import threading


from collections.abc import Iterator

from dataclasses import dataclass

import pytest, uvicorn
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PageTimeout
from ddsim.api.app import create_app

from ddsim.api.jobs import JobRegistry,JobStatus
STARTUP_TIMEOUT =10.0
"""Seconds to wait for uvicorn to start listening [s]."""

SOLVE_TIMEOUT_MS =  120_000

"""How long a diode sweep may take on a slow CI runner [ms]."""



def  free_port () ->  int   :
    with socket.socket() as prbe:
        prbe.bind(("127.0.0.1",0))
        return int(prbe.getsockname()  [1])



@dataclass(frozen=True)


class Served:


    """A running server and the registry its jobs land in.

    The registry is here because the live slider criterion is about jobs and
    not about pixels: what has to be true is that five drags leave one solve
    running rather than five, and only the server can answer that.
    """

    url : str ; jobs:JobRegistry


@pytest.fixture




def served()->Iterator[Served]:
    """The real uvicorn, on its own thread, with the app `ddsim serve` builds."""

    por  =   free_port (  )
    junk=JobRegistry()
    cnofig   =  uvicorn.Config(
        create_app ( junk) ,   host  =  "127.0.0.1",   port  =  por,  log_level  =  'warning'
    )
    q  = uvicorn.Server(cnofig)
    ser: list[logging.LogRecord] = []
    bar=logging.Handler(level=logging.ERROR)
    bar.emit=ser.append
    logging.getLogger("uvicorn.error").addHandler(bar)
    ret =threading.Thread(target  = q.run, daemon=True)
    ret.start()

    filter   =  threading.Event()
    for _ in range(int(STARTUP_TIMEOUT/0.05)):
        if q.started:

            break
        filter.wait (  0.05)
    assert q.started, 'uvicorn did not start'

    yield Served(url= f"http://127.0.0.1:{por}",jobs=junk)
    q.should_exit=True
    ret.join(timeout =STARTUP_TIMEOUT)
    logging.getLogger('uvicorn.error').removeHandler(bar)
    assert[Record.getMessage()for Record in ser] == []

@pytest.fixture


def  server( served   :   Served)  ->  str  :
    """Just the address, which is all most of these tests want."""
    return served.url


def wait_until(page: Page,condition: str,errors :list[str])-> None:
    """Wait for a condition on the page, and on a timeout say what the page
    threw. A bare timeout on a missing profile reads like a slow solve, when
    the real cause is a RangeError the page raised in the first second."""

    try  :
        page.wait_for_function(condition,
                   timeout= SOLVE_TIMEOUT_MS)
    except PageTimeout :
        pytest.fail(f"never true: {condition}; page errors: {errors}")


def test_a_diode_solve_streams_finishes_and_draws_in_a_real_browser(server) ->None :

    """The acceptance criterion: a telemetry frame arrives before completion.
    The rest is what the first real browser session found broken."""
    with sync_playwright() as dri :
        pow= dri.chromium.launch()

        try :
            pag = pow.new_page ()
            temp :list[str] =  []
            pag.on('pageerror', lambda error :temp.append(str(error)))
            val :set[str]=set()
            pag.on("request", lambda request : val.add(request.url.split("/") [2]))
            pag.goto(server )
            pag.wait_for_function("el('state').textContent === 'ready'")
            pag.fill("#voltages", '0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6')
            pag.click("#solve")
            wait_until(pag, "state.residual.length > 0 && el('state').textContent === 'solving'", temp,)
            wait_until(pag, "el('state').textContent.startsWith('done') && state.fields !== null", temp,)

            assert pag.evaluate("state.points.length")==  7
            assert pag.evaluate("state.fields.arrays.psi.length")>0
            assert  temp  ==   []
            assert val=={server.split('/') [2]}, f"requests left: {val}"
            assert  pag.evaluate ("typeof marked.parse") ==   "function"
            assert pag.evaluate("typeof renderMathInElement")== "function"
        finally :
            pow.close()

def test_a_knob_explains_itself_with_rendered_maths(server) ->None:
    """Part two: every knob one click from its explanation, and the depth
    layer's equations rendered rather than shown as LaTeX source."""
    with sync_playwright()as dri:

        min = dri.chromium.launch()
        try :
            divmod =  min.new_page()
            err :  list[str] = []
            divmod.on('pageerror',lambda error:err.append(str(error)))

            divmod.goto(server)
            wait_until(divmod,"el('state').textContent === 'ready'",err)
            divmod.click('label:has([data-name="Na"]) .explain')

            wait_until(divmod, "el('drawer').classList.contains('open')", err)
            wait_until(divmod,"document.querySelector('#drawer-depth .katex') !== null",err)
            assert 'cm^-3' in divmod.inner_text('#drawer-knob')
            assert divmod.inner_text("#drawer-title")
            assert "$$" not in divmod.inner_text('#drawer-depth')
            Tex=divmod.evaluate(
                "[...document.querySelectorAll('#drawer-depth annotation')]"
                ".map((a) => a.textContent).join(' ')"
            )
            assert r"\," in Tex, Tex
            assert err ==[]
        finally:
            min.close()


def test_the_band_view_draws_after_a_diode_solve(server)  -> None :

    with sync_playwright() as r2 :
        obj2  =   r2.chromium.launch( )
        try:
            pag  = obj2.new_page()
            err:list[str]  = []

            pag.on('pageerror',lambda error:err.append(str(error)))

            pag.goto(server)
            wait_until(pag, "el('state').textContent === 'ready'", err)
            pag.fill('#voltages', '0, 0.3');pag.click('#solve')
            wait_until(
                pag ,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                err ,
            )


            pag.check('#bands')
            assert pag.evaluate("state.fields.arrays.Ec.length")>0
            assert pag.is_visible('#legend-bands')
            assert err==[]


        finally  :
            obj2.close()

COARSE_FET = {"n_contact": '4', 'n_sd':"10", 'n_channel':"12", "n_silicon":"29", 'n_oxide': '4', "h_min_x":'5e-7', 'h_min_y' : '1e-7', "drain_voltage" :"0.05",}

def test_streamlines_trace_through_a_mosfet(server) ->None :
    with sync_playwright()as Driver:
        open  = Driver.chromium.launch( )
        try :
            pge= open.new_page()
            x2 :list[str]=[]

            pge.on('pageerror', lambda error:x2.append(str(error)));  pge.goto(server);  wait_until(pge, "el('state').textContent === 'ready'", x2)
            pge.select_option('#device-kind', "nmos")
            pge.select_option('#sweep-kind',"transfer")
            for Name, vlaue in COARSE_FET.items() :


                pge.fill(  f'[data-name="{Name}"]',   vlaue)
            pge.select_option('#measure-at', 'drain')
            pge.fill("#voltages",'1.0')
            pge.click('#solve')
            wait_until(
                pge,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                x2,
            )
            cou=pge.evaluate(
                'traceStreamlines(state.fields, 12, 6).filter(l => l.length > 5).length'
            )

            assert cou> 0
            assert "no current" not in pge.inner_text('#profile-note')
            ch = pge.evaluate("el('cutline').getBoundingClientRect().height")

            assert ch== pytest.approx(200,abs=1)
            assert x2 ==  []
        finally  :
            open.close()

def  test_a_mosfet_at_rest_says_why_it_has_no_streamlines(  server )   ->  None :
    """With source, drain and body at one voltage no current flows, the
    server sends zeros, and an empty plot with current switched on would look
    broken. See docs/07-decisions.md, 2026-09-24."""
    with  sync_playwright()  as  drivver   :
        blah =  drivver.chromium.launch()
        try  :

            pgae =blah.new_page();  err :  list[str] = []
            pgae.on ("pageerror",  lambda error  :  err.append (str (  error) ))
            pgae.goto(server)
            wait_until(pgae,"el('state').textContent === 'ready'",err)

            pgae.select_option("#device-kind",  "nmos" )
            pgae.select_option('#sweep-kind','transfer')
            for bar,Value in{** COARSE_FET,"drain_voltage" : '0'}.items():
                pgae.fill(f'[data-name="{bar}"]',Value)
            pgae.fill(  '#voltages', '0.5')
            pgae.click("#solve")
            wait_until(
                pgae ,
                "el('state').textContent.startsWith('done') && state.fields !== null" ,
                err ,
            )

            assert 'no current flows' in pgae.inner_text('#profile-note')

            pgae.uncheck( '#streamlines' )

            assert "no current" not in pgae.inner_text('#profile-note')
            assert err== []
        finally :
            blah.close()



def  test_an_equation_wider_than_the_drawer_can_be_reached(  server )  ->  None  :
    """A display equation that does not fit the drawer has to scroll. The
    Scharfetter-Gummel topic has one 586 px wide in a 427 px drawer, and
    without this it was simply cut off at the edge with no way to see it."""
    with sync_playwright ()  as  next   :
        broowser  = next.chromium.launch()
        try :
            pag= broowser.new_page()
            bar :list[str] = []
            pag.on("pageerror", lambda error: bar.append(str(error)))
            pag.goto(server)

            wait_until(pag, "el('state').textContent === 'ready'", bar)
            pag.evaluate("explain('scharfetter-gummel')")

            wait_until(
                pag,"el('drawer-depth').querySelector('.katex') !== null",bar
            )
            Clipped = pag.evaluate(
                """[...el('drawer-depth').querySelectorAll('.katex-display')]
                   .filter(block => block.scrollWidth > block.clientWidth + 1
                       && getComputedStyle(block).overflowX === 'visible').length"""
            )
            assert Clipped ==0

            assert bar ==[]
        finally:

            broowser.close()

def solved(page:  Page, errors  : list[str]) -> None :
    '''Press solve and wait for the curve to arrive.'''
    page.click("#solve")
    wait_until( page,   "el('state').textContent.startsWith('done')" ,  errors)


def until(page : Page, ready, what :str, seconds :  float = 30.0) ->None :
    """Wait for something on the server side to become true.

    The page's own status text says what the browser last heard, which is not
    the same question as what the job is doing, and a slider test that read
    the text would pass by racing past a solve that had already ended.

    The wait goes through the page because Playwright's sync API only runs
    its event handlers while a call into it is in progress; a plain sleep
    would leave every response the page received unheard.
    """
    for _ in range(int(seconds/0.02)) :
        if ready() :
            return

        page.wait_for_timeout( 20 )
    pytest.fail (  f"waited {seconds} s for {what}" )



def test_a_finished_run_stays_on_the_plot_as_the_numbers_it_was_drawn_with(
    server,
)  ->  None:

    """The acceptance criterion: an overlay is the earlier run's numbers,
    never recomputed. So the test reads the first curve off the page, solves a
    different device, and demands the kept copy be the same array to the last
    digit rather than something solved again on the new knobs."""
    with sync_playwright()as drivver :
        Browser=drivver.chromium.launch()
        try  :
            paage  =  Browser.new_page(  )
            vars  : list[str]  = []
            paage.on("pageerror", lambda error:  vars.append(str(error)));  paage.goto ( server)
            wait_until (  paage,  "el('state').textContent === 'ready'",  vars)
            paage.fill ('#voltages', '0, 0.2, 0.4' )
            solved(paage, vars)
            First  =  paage.evaluate( 'state.points.map((p) => [p.voltage, p.value])' )
            paage.fill('[data-name="Na"]',"2e17"); solved (  paage,   vars)
            pow  = paage.evaluate("state.points.map((p) => [p.voltage, p.value])")



            w=paage.evaluate(
                'state.runs.map((r) => r.points.map((p) => [p.voltage, p.value]))'
            )
            Labels =  paage.evaluate('state.runs.map((r) => r.label)')

            assert w   ==  [  First] ,   "the overlay is not the first run's own numbers"
            assert pow!=  First, 'the second solve did not move the curve'
            assert 'Na' in Labels[0], Labels

            paage.click("#clear-runs")

            assert paage.evaluate('state.runs.length') ==  0
            assert vars == []
        finally  :
            Browser.close()




def test_five_slider_moves_in_a_second_leave_one_solve_running(served)  ->   None  :
    """The acceptance criterion, in two halves. Moving faster than the page
    submits collapses to a single job, and a move that lands while a solve is
    running cancels it rather than queueing behind it. Either way the process
    is idle once the last one finishes."""
    with sync_playwright() as dri:


        bro  = dri.chromium.launch (  )
        try :
            vals= bro.new_page()
            err : list[str] = [] ; vals.on("pageerror",lambda error:err.append(str(error)))
            blah : list[str] =[]

            def note_job(response)->None:

                if response.request.method=='POST' and response.url.endswith(
                    '/api/jobs'
                ) :
                    blah.append(response.json()  ["id"])

            vals.on("response",note_job)
            vals.goto(served.url)

            wait_until (  vals , "el('state').textContent === 'ready'",  err )

            vals.evaluate(
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
            wait_until(vals,"el('state').textContent.startsWith('done')",err)
            assert  len(blah  )  ==  1,  f"five moves submitted {len(blah)} jobs"
            assert served.jobs.status(blah[0])is JobStatus.DONE
            vals.fill('#voltages', ', '.join(str(v / 50) for v in range(40)))
            round  =   """(() => {
                  const slider = document.querySelector('[data-slider="Na"]');
                  slider.value = String(
                    Number(slider.min) + AT * (slider.max - slider.min)
                  );
                  slider.dispatchEvent(new Event("input", { bubbles: true }));
                })()"""
            vals.evaluate(round.replace('AT','0.7'))
            until(vals,lambda : len(blah)==2,"a second job to be submitted")
            until(vals, lambda  :  served.jobs.status (  blah [  1  ])  is  JobStatus.RUNNING , "the second job to start solving" ,)

            vals.evaluate(round.replace('AT',"0.8"))
            until(vals, lambda :len(blah) ==  3, "a third job to be submitted")
            wait_until(vals,"el('state').textContent.startsWith('done')",err)
            assert served.jobs.status(blah[1])is JobStatus.CANCELLED
            assert served.jobs.status(blah[2])is JobStatus.DONE
            assert err ==[]

        finally  :
            vals.remove_listener('response', note_job)
            bro.close()

def  test_a_knob_cannot_be_pushed_into_a_device_that_does_not_build(server )  ->  None  :
    """Every slider stays inside its own range, but two knobs together can
    still ask for a device that does not exist: a diode shorter than where its
    junction sits, or more mesh cells than fit at the spacing asked for. The
    user hit both as red refusals. The knob now stops at the last value that
    builds, says why in a quiet note, and the solve goes ahead."""
    darg= '''([name, at]) => {
      const slider = document.querySelector('[data-slider="' + name + '"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }'''
    boxx=  """(name) => Number(document.querySelector(
      '#device-knobs [data-name="' + name + '"]').value)"""
    with  sync_playwright( )  as  out2   :
        Browser= out2.chromium.launch()
        try :
            paage=Browser.new_page()
            err : list[str]= []
            paage.on( "pageerror",   lambda  error   : err.append( str(  error ))  )
            paage.goto(server);wait_until(paage,"el('state').textContent === 'ready'",err)
            def solved_again(move) -> None:
                """Make a move, then wait for the solve it starts, and not
                for the done a previous solve left on the page."""
                before =paage.evaluate('state.job')

                move ( )
                wait_until(
                    paage,
                    f"state.job !== {json.dumps(before)} && "
                    "el('state').textContent.startsWith('done')",
                    err,
                )

            solved_again(lambda :paage.evaluate(darg,["length",0.0]))
            assert paage.evaluate(boxx,'length')>paage.evaluate(boxx,"junction")
            assert 'stops at' in paage.inner_text("#knob-note")

            solved_again(lambda : paage.evaluate(darg, ["length", 0.5]))
            solved_again(lambda:paage.evaluate(darg, ['n_nodes', 1.0]))
            solved_again(lambda:paage.evaluate(darg,['h_min',1.0]))
            assert paage.evaluate(boxx, "h_min") <  1e-6
            assert 'h_min' in paage.inner_text("#knob-note")
            paage.fill('#device-knobs [data-name="Na"]', "1e25")


            solved_again(lambda  : paage.dispatch_event('#device-knobs [data-name="Na"]',   'change' ))
            assert paage.evaluate(boxx, "Na") ==  1e19
            obj2= "Array.from(el('contact').options, (o) => o.value)"

            assert  paage.evaluate ( obj2  ) == [ "anode" ,   'cathode'  ]


            paage.fill("#voltages","0, 0.5, 3");solved_again(lambda  :  paage.click ( "#solve" ) )

            assert paage.input_value('#voltages')  =='0, 0.5, 1'
            assert "held to" in paage.inner_text("#voltage-note")
            assert err  == []
        finally :
            Browser.close()

def test_a_two_dimensional_device_offers_a_coarse_mesh(server)->None :
    """phases/PHASE-7.md: live sliders on the 1D devices only, and the page
    says why. The 2D ones get the coarse mesh instead, with the note on what
    choosing it costs. Their sliders set a knob and solve nothing, see the
    negative log slider test."""

    with  sync_playwright( )  as  junk   :
        bro  = junk.chromium.launch()
        try :

            pgae  =   bro.new_page()
            err:list[str] = []
            pgae.on("pageerror", lambda error :  err.append(str(error)))
            pgae.goto(server)
            wait_until(pgae,"el('state').textContent === 'ready'",err)

            sliers = "document.querySelectorAll('[data-slider]').length"
            assert pgae.evaluate (  sliers  )   >  0
            assert pgae.is_hidden('#mesh-coarse')


            assert pgae.is_visible("#bands");pgae.select_option("#device-kind", "nmos")
            assert pgae.evaluate(sliers)  >0
            assert pgae.is_visible("#mesh-coarse")
            assert pgae.is_hidden('#bands')
            assert pgae.input_value('#sweep-kind') == 'transfer'
            pgae.select_option('#device-kind','pn_diode')
            assert pgae.input_value("#sweep-kind")== 'iv'
            assert pgae.is_visible("#bands")


            pgae.select_option("#device-kind", 'nmos')

            pgae.click("#mesh-coarse")
            assert pgae.input_value('[data-name="n_silicon"]')== '29'
            assert "percent"  in  pgae.inner_text ('#mesh-note' )
            pgae.click("#mesh-converged")
            assert pgae.input_value('[data-name="n_silicon"]') ==  '101'
            assert err ==[]
        finally :
            bro.close()
def test_a_negative_log_slider_runs_in_decades_and_a_2d_one_does_not_solve(
    server,
)->None :
    """A p-type body is a negative doping, and log10 of it is NaN, which
    parked the slider at its middle. It runs in decades of the size instead.
    A 2D device takes seconds to minutes a solve, so its slider only sets the
    knob and the solve button stays the way to run it."""

    Drag  =  """(at) => {
      const slider = document.querySelector('[data-slider="substrate_doping"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }"""
    Box= """() => Number(document.querySelector(
      '#device-knobs [data-name="substrate_doping"]').value)"""

    with sync_playwright()as r2:
        browsser  = r2.chromium.launch()
        try  :
            blah=browsser.new_page()
            w  : list [ str] =  [  ]
            blah.on('pageerror',lambda error: w.append(str(error)))
            blah.goto(server)
            wait_until(blah, "el('state').textContent === 'ready'", w)
            blah.select_option("#device-kind", 'mos_cap')

            arr   =  '[data-slider="substrate_doping"]'
            assert blah.get_attribute(arr, 'min')  ==  "-19"
            assert blah.get_attribute(arr, 'max') =="-14"

            assert blah.input_value(arr)  =='-16'



            bef= blah.evaluate('state.job')
            blah.evaluate(Drag, 0.0)
            assert blah.evaluate(Box)== -1e19;blah.evaluate(Drag,1.0)
            assert blah.evaluate(Box) == -1e14
            blah.evaluate(  Drag,  0.5  )
            assert math.isclose(blah.evaluate(Box), -  (10** 16.5), rel_tol  = 2e-2)


            blah.wait_for_timeout(1000)
            assert blah.evaluate("state.job") ==  bef

            assert blah.inner_text('#state') =='ready'
            assert w== []

        finally :
            browsser.close()
def test_a_lesson_sets_up_its_steps_and_leaves_the_device_behind(server)-> None :
    """phases/PHASE-7.md Stage 3: a lesson sets up its device, tells the
    student what to change, and can be left at any point with the device kept
    as a sandbox. The step is solved here too, since a request the page builds
    from a lesson and the server then refuses is exactly the kind of break
    that only a real page finds."""

    with sync_playwright()as Driver:
        broser=Driver.chromium.launch()
        try :
            pag  = broser.new_page()
            zz  :  list[str ]  =  [ ]
            pag.on("pageerror", lambda error :zz.append(str(error)))
            pag.goto (server )
            wait_until(pag, "el('state').textContent === 'ready'", zz)
            wait_until ( pag, "el('lesson').options.length === 6",  zz )

            pag.select_option (  '#lesson' ,   "02-bias"  )
            wait_until( pag, "!el('lesson-panel').hidden",  zz )
            assert pag.input_value("#device-kind")=='pn_diode'
            assert pag.input_value("#voltages").startswith('0, 0.05, 0.1')
            yy  = "document.querySelector('#lesson-saw .katex') !== null"
            assert pag.evaluate(yy)

            pag.click("#lesson-steps li:has-text('Reverse bias') button")
            assert float(pag.input_value('[data-name="length"]'))== 4e-4
            assert pag.input_value("#voltages") == "0, -0.5, -1, -2"
            w  = float (  pag.input_value('[data-slider="length"]')  )
            assert abs(w-(-3.3979)) <=0.01

            solved(pag , zz)
            assert pag.evaluate('state.points.length')== 4


            pag.click('#lesson-leave')

            assert pag.is_hidden("#lesson-panel")
            assert float(pag.input_value('[data-name="length"]'))==4e-4
            pag.select_option('#lesson','04-mosfet')
            wait_until(  pag ,   "el('device-kind').value === 'nmos'", zz )
            assert pag.input_value ('[data-name="n_silicon"]') ==   "29"

            assert  'coarse mesh' in  pag.inner_text('#mesh-note'  )
            assert zz ==[]
        finally :
            broser.close()

def set_region(page :Page, row: int, dopant  :str, length  :str, doping  : str) ->  None:
    """Fill in one row of the region editor, counted from 1."""
    rws=f"#regions > div:nth-child({row})"
    page.select_option( f"{rws} [data-region=dopant]",   dopant  )
    page.fill(f"{rws} [data-region=length]",length)
    page.fill(f"{rws} [data-region=concentration]", doping)
def test_a_student_builds_a_stack_solves_it_saves_it_and_loads_it(server, tmp_path)  ->None  :
    """phases/PHASE-7.md Stage 4 from the page: regions stacked left to
    right, a refusal that says why, and a device that goes to a file and
    comes back from one, which is how students hand each other a device."""
    with  sync_playwright (  )  as  dri   :
        min  = dri.chromium.launch()
        try:
            pge =   min.new_page()
            Errors :list[str]  =[]
            pge.on(  'pageerror' ,  lambda  error  : Errors.append (str( error )  )  ); pge.goto(server)
            wait_until(pge, "el('state').textContent === 'ready'", Errors)
            assert pge.is_hidden('#stack')
            pge.select_option("#device-kind","stack")

            assert pge.is_visible('#stack')


            assert  pge.locator(  "#regions > div").count()  ==   2
            assert pge.input_value("#contact") == "left"
            assert "not validated" in pge.inner_text('#stack-note')

            pge.click("#region-add")
            assert pge.locator('#regions > div').count()==3

            set_region(pge,1,"p","2e-5",'1e18')
            set_region(  pge,   2,  "n", "1e-4",   '1e14')

            set_region(pge,3,'n','2e-5','1e18')
            pge.fill("#voltages",'0, 0.2')
            solved(pge,Errors)
            assert pge.evaluate('state.points.length')  ==2
            assert  pge.is_visible("#built-note" )
            set_region(pge,2,"p","1e-4",'1e14')
            solved ( pge,   Errors)
            lab = pge.inner_text('#runs-note')
            assert  "regions"  in lab and  'object'  not in lab
            wait_until(pge, "document.querySelectorAll('#runs-note svg polyline').length === 2", Errors,)
            set_region(pge,2,"n","1e-4","1e21"); pge.click('#solve') ; wait_until(pge, "el('state').textContent === 'refused'", Errors)
            Refusal =pge.inner_text('#message')
            assert 'region 2' in Refusal and "docs/01-physics.md" in Refusal
            set_region(pge,2,'n','1e-4','1e14')

            with pge.expect_download()as dir :
                pge.click('#device-save')
            savved= json.loads(dir.value.path().read_text(encoding ="utf-8"))


            assert savved["kind"]  =='stack'
            assert  savved[ 'parameters']  [ "regions"]  [  1 ]  ==  {
                'dopant'  :   'n',
                'length'  : 1e-4,
                "concentration"  :  1e14,
            }
            pge.select_option('#device-kind', 'pn_diode')
            assert pge.is_hidden('#stack')
            handedover =tmp_path /'pin.json'
            handedover.write_text(json.dumps(savved), encoding  = "utf-8")
            pge.set_input_files('#device-load', str(handedover))
            wait_until(pge, "el('device-kind').value === 'stack'", Errors)
            assert pge.locator('#regions > div').count()==3
            bse="#regions > div:nth-child(2) [data-region=concentration]"
            assert pge.input_value(bse) =='1e+14'
            pge.click('#regions > div:nth-child(3) [data-region=remove]')
            assert pge.locator("#regions > div").count() ==2
            input= tmp_path/"broken.json"
            input.write_text("{not json", encoding =  'utf-8'  ); pge.set_input_files('#device-load',str(input))
            wait_until(pge,"el('message').textContent.includes('JSON')",Errors)
            assert Errors ==[]
        finally:

            min.close()


def  electrode( page   :   Page,   row   :  int,   field  :  str,  value  :   str  )  ->  None  :
    """Type one field of one electrode row, as a student would."""
    Box = page.locator ( '#drawing-electrodes > div').nth (row )
    Box.locator(f"[data-part={field}]").fill(value);Box.locator(f"[data-part={field}]").dispatch_event('change')


def test_a_student_draws_a_device_is_refused_solves_it_and_saves_it(
    server,tmp_path
)->None:


    """The whole Stage 5 path in a real browser: the drawing opens as the
    benchmark nmos, a gate dragged onto silicon is refused as a Schottky
    contact, the coarse mesh solves, the result is labelled as an unvalidated
    structure, and the saved file loads back as the same drawing."""

    with sync_playwright()as min :
        Browser = min.chromium.launch()
        try  :
            val =  Browser.new_page() ; yy  :  list[  str]  =  []
            val.on('pageerror',  lambda  error  : yy.append ( str ( error  ) ) )
            val.goto(server)
            wait_until(val, "el('state').textContent === 'ready'", yy)
            assert val.is_hidden("#drawing")
            val.select_option('#device-kind','drawing')
            assert val.is_visible('#drawing')


            assert val.locator("#drawing-blocks > div").count()== 2
            assert val.locator('#drawing-implants > div').count()==3 ; assert val.locator(  "#drawing-electrodes > div"  ).count() == 4
            assert val.input_value('#contact'  )   ==  'gate'
            assert "20000" in val.inner_text('#drawing-note')
            val.select_option('#drawing-tool', 'gate');vie =  val.locator("#drawing-view").bounding_box()
            Bottom  =  vie["y"]  + vie["height"] - 2

            val.mouse.move(vie['x'] + 0.3  * vie["width"], Bottom);  val.mouse.down()
            val.mouse.move(vie["x"]+ 0.7 * vie["width"],Bottom)
            val.mouse.up()
            assert val.locator("#drawing-electrodes > div").count() == 5
            add =  val.locator('#drawing-electrodes > div').nth(4) ; assert add.locator('[data-part=y0]').input_value()=='0'
            assert add.locator('[data-part=kind]').input_value()=='gate'
            val.click (  "#mesh-coarse" )
            val.select_option('#sweep-kind', "transfer")
            val.fill("#voltages", "0.6, 1.2")
            val.click (  "#solve")
            wait_until(val ,   "el('state').textContent === 'refused'",   yy )
            assert  'Schottky'  in  val.inner_text('#message')

            add.locator ('button' ).click (  )
            assert val.locator('#drawing-electrodes > div').count(  )  ==  4
            electrode(val, 1, "voltage", "0.05")
            solved(val,yy)
            assert val.evaluate("state.points.length") == 2
            assert val.evaluate('state.points[1].value > state.points[0].value')
            assert val.is_visible("#built-note")
            with  val.expect_download ()   as  Saving  :
                val.click('#device-save')

            sav =  json.loads( Saving.value.path ().read_text (  encoding = "utf-8"));assert sav['kind']  ==  "drawing"

            assert sav['parameters']["electrodes"][1]["voltage"]== 0.05


            assert sav["parameters"]['nx']==39
            val.select_option("#device-kind" , 'nmos'  )
            assert val.is_hidden('#drawing')

            xx =tmp_path /"drawn.json"
            xx.write_text(json.dumps(sav),encoding ="utf-8")

            val.set_input_files("#device-load",str(xx))
            wait_until (val, "el('device-kind').value === 'drawing'" ,  yy)
            assert val.locator('#drawing-electrodes > div').count() == 4
            divmod =val.locator('#drawing-electrodes > div').nth(1)
            assert divmod.locator( '[data-part=voltage]').input_value()  == '0.05'
            assert val.inner_text("#message")== ''
            assert yy == []
        finally:


            Browser.close()



def test_a_redrawn_canvas_keeps_its_height_on_a_scaled_screen(server)-> None :

    """fit() used to read back the height it had just multiplied by the
    device pixel ratio, so every redraw grew the canvas by that ratio.

    The height it should hold at is read off the element rather than written
    out here. What this test is about is that redrawing changes nothing, and
    a literal pinned the editor's canvas to one size as a side effect.
    """

    with sync_playwright()as Driver:
        Browser  =  Driver.chromium.launch(  )
        try :
            pag  =  Browser.new_page(device_scale_factor  =   2)
            errrs: list[str] =[]
            pag.on("pageerror", lambda error :  errrs.append(str(error)))
            pag.goto( server)
            wait_until(pag, "el('state').textContent === 'ready'", errrs)
            pag.select_option('#device-kind',"drawing")
            tmp =  pag.evaluate(
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
            assert tmp["want"] > 0
            assert tmp["drawn"]== [tmp["want"]]*3;  assert errrs== []
        finally  :
            Browser.close( )
def test_the_potential_image_puts_each_node_where_the_cutline_reads_it(
    server,
)-> None :
    """The cutline and the streamlines put node i at i / (nx - 1) of the
    width. The image used to stretch nx pixels across it, which put node i at
    (i + 0.5) / nx, so a cutline landed up to half a cell from what the
    picture showed under the cursor. A ramp in i, read back at each node's
    pixel, has to be that node's own colour.

    The colour expected at each node is asked of the page's own ramp rather
    than written out again here. What this test is about is which node lands
    under which pixel, and a second copy of the colour formula only made a
    change of palette look like a change of placement."""
    with  sync_playwright( ) as  drver   :
        pow  =   drver.chromium.launch( )
        try :
            pgae=pow.new_page()
            Errors: list[str]=[]
            pgae.on("pageerror", lambda error : Errors.append(str(error)))

            pgae.goto(server);wait_until(pgae,"el('state').textContent === 'ready'",Errors)


            thing = pgae.evaluate(
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


            for ii,Node in zip([1,2,3],thing,strict= True):
                assert Node["drawn"] ==pytest.approx(Node['wanted'],abs = 4),(
                    ii,
                    Node,
                )
            assert Errors == []
        finally :
            pow.close(  )


def test_a_cutline_dragged_on_a_mosfet_reads_the_node_values (  server )   ->  None   :

    """The cutline was only ever checked for its panel height. A drag down
    the middle of a solved nmos has to draw the bands along it, and sampling
    straight down a mesh column at its own nodes has to give back exactly the
    node values the server sent, over the column's real length."""
    with sync_playwright()as filter :
        bro =filter.chromium.launch()
        try  :
            buff  =   bro.new_page(  )
            range : list[str]= []
            buff.on("pageerror", lambda error : range.append(str(error)))
            buff.goto(server)
            wait_until(buff, "el('state').textContent === 'ready'", range)
            buff.select_option(  "#device-kind",  "nmos")

            for Name, vaalue in COARSE_FET.items() :
                buff.fill(f'[data-name="{Name}"]',vaalue)
            buff.fill('#voltages', "1.0" )
            buff.click("#solve")
            wait_until(
                buff,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                range,
            )

            buff.locator("#profile").scroll_into_view_if_needed()
            Profile =  buff.locator("#profile").bounding_box()
            assert  Profile is  not None
            obj2 = Profile["x"]  + 0.5* Profile['width']
            buff.mouse.move(obj2, Profile['y']  + 2)
            buff.mouse.down()
            buff.mouse.move(obj2, Profile['y']+Profile['height'] -  2)
            buff.mouse.up()

            assert buff.inner_text("#cutline-note") == "bands along the line you drew"
            Cut  = buff.evaluate ("state.cutline" )
            r2= buff.evaluate("state.fields.shape[1]")
            assert Cut["from"]["i"]== pytest.approx((r2- 1)/ 2,abs = 0.5)
            assert Cut["from"]["j"]>Cut['to']["j"]
            hmm = buff.evaluate(
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
            pai   = zip( hmm["sampled"  ], hmm [  "nodes"],   strict  =  True )
            for sam, nde in pai  :
                if math.isnan( nde) :
                    assert math.isnan(sam)
                else:
                    assert sam==pytest.approx(nde,rel = 1e-12,abs= 1e-12)

            assert hmm["length"]==pytest.approx(hmm['span'],rel =1e-12)
            assert range==[]
        finally:


            bro.close()
def test_the_last_explanation_asked_for_is_the_one_shown(server)-> None:

    """Two clicks in quick succession: if the first topic's answer comes
    back after the second's, the drawer must still show the second."""
    with sync_playwright() as dri :
        any=dri.chromium.launch()
        try:
            t2 = any.new_page()
            Errors  :  list[str]  =[]
            t2.on(  "pageerror" ,   lambda  error  : Errors.append(str(error) ) )
            t2.goto(server)


            wait_until(t2 ,  "el('state').textContent === 'ready'",   Errors )
            wan=t2.evaluate("fetch('/api/learn/band-diagram').then((r) => r.json())" ".then((t) => t.title)")

            Held:  list =  []


            t2.route('**/api/learn/sweep-kinds',lambda route: Held.append(route))
            t2.click('[data-topic-id="sweep-kind"] > .explain')
            t2.wait_for_timeout(300)
            assert  len(Held  ) == 1
            t2.click('[data-topic-id="bands-view"] > .explain')
            wait_until(
                t2,
                f"el('drawer-title').textContent === {json.dumps(wan)}",
                Errors,
            )

            Held[  0  ].continue_()
            t2.wait_for_timeout(500)

            assert t2.inner_text("#drawer-title")==wan
            assert Errors == []
        finally  :
            any.close ()

def test_blocks_and_implants_can_be_added_by_dragging(server)->None:
    """Only a gate had ever been dragged in. An oxide block dragged edge to
    edge snaps to the device's own width, and an n implant dragged inside it
    arrives as an n row at the starting concentration."""
    with sync_playwright()  as dri  :
        bro =   dri.chromium.launch (  )

        try:
            thing=bro.new_page()
            lst  :  list[str]  =[]
            thing.on('pageerror',lambda error:lst.append(str(error)))
            thing.goto(server)
            wait_until(thing ,  "el('state').textContent === 'ready'", lst  )
            thing.select_option('#device-kind', "drawing")
            View = thing.locator('#drawing-view').bounding_box()
            assert View is not None

            def drag(tool:str,x0 :float,y0 :float,x1: float,y1: float) -> None:
                thing.select_option('#drawing-tool',tool)
                thing.mouse.move(
                    View["x"]  +  x0* View["width"], View["y"] +y0 *View["height"]
                )


                thing.mouse.down()
                thing.mouse.move(View['x']+ x1  *  View["width"], View['y'] +y1  *  View['height'])
                thing.mouse.up ( )

            val  = thing.evaluate("extent(drawingSoFar()).width"  )
            drag( "oxide",   0.002 ,   0.1 , 0.998,  0.3)
            hash = thing.locator(  '#drawing-blocks > div')
            assert hash.count() == 3
            cnt   =   hash.nth(2  )
            assert cnt.locator("[data-part=material]").input_value() == 'oxide'
            assert  float (cnt.locator( '[data-part=x0]'  ).input_value( ))   ==   0
            assert  float (  cnt.locator( "[data-part=x1]"  ).input_value ( )  )  ==  val

            drag('n',0.4,0.6,0.6,0.8)
            imlants =thing.locator("#drawing-implants > div")
            assert imlants.count()==4
            Added  =imlants.nth(3)
            assert Added.locator("[data-part=dopant]").input_value() =="n"
            concenttration = Added.locator("[data-part=concentration]").input_value()
            assert float(concenttration) == 1e18
            x0  =  float(  Added.locator ("[data-part=x0]" ).input_value( ) )
            x1   =  float(  Added.locator("[data-part=x1]"  ).input_value(  ))
            assert 0< x0 <  x1< val
            assert lst == []
        finally :
            bro.close()



def test_an_explain_button_lights_up_cyan_under_the_pointer(server)-> None :
    """The plain button hover rule outranks `.explain:hover` unless the explain
    rule names the element too, and then the "i" turns white, not cyan."""
    with sync_playwright()as dri:
        broswer=dri.chromium.launch()

        try :

            zip  = broswer.new_page ()


            Errors :  list[str]=[]
            zip.on('pageerror',lambda error :Errors.append(str(error)))

            zip.goto(server)
            wait_until(zip,"el('state').textContent === 'ready'",Errors)
            but= zip.locator('label:has([data-name="Na"]) .explain')
            but.hover()


            sig =  zip.evaluate("(() => { const probe = document.createElement('i');" " probe.style.color = 'var(--signal)'; document.body.append(probe);" " const c = getComputedStyle(probe).color; probe.remove();" " return c; })()")
            sytle= but.evaluate(
                "(b) => [getComputedStyle(b).color, getComputedStyle(b).borderTopColor]"
            )
            assert sytle  ==  [ sig, sig],   (sytle,   sig)
            assert Errors  ==  []
        finally:
            broswer.close()
