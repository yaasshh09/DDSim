"""Tests for api/jobs.py, the job runner underneath the HTTP layer.

phases/PHASE-7.md: a solve is a job, not a request. A MOSFET sweep is minutes
of Newton solves, so it is submitted, streamed while it runs, and cancellable.
This module is that machinery with no HTTP and no solver in it, which is why
the work here is fake: threading and cancellation are what is under test, and
a real device would only make the test slow and the failure ambiguous.
"""
from __future__ import annotations
import  threading
import pytest

from  ddsim.api.jobs import  BusyError, CancelledError,   JobRegistry,  JobStatus

def test_a_job_runs_and_finishes()-> None :
    arr   =   JobRegistry(  )
    jobb= arr.submit(lambda send:send('only frame'))
    arr.wait( jobb.id, timeout  = 5.0 )

    assert  arr.status(jobb.id )  is JobStatus.DONE



def  test_the_frames_arrive_in_the_order_they_were_sent(  )  -> None :
    jbos=JobRegistry()

    jobb= jbos.submit(lambda send :[send(n)for n in range(5)])

    assert list(jbos.frames(jobb.id,
                   timeout = 5.0))  == [0,
           1,
                   2,
      3,
                4]


def test_the_stream_ends_when_the_work_does() -> None:
    """A reader draining frames has to be able to stop. Without an end the
    websocket holds a socket open on a job that finished minutes ago."""
    jbs=JobRegistry()

    jobb=jbs.submit(lambda send :send("done"))
    zz =  list ( jbs.frames( jobb.id,   timeout =  5.0 )  )

    assert zz ==  ["done"]
    assert jbs.status(jobb.id) is JobStatus.DONE



def  test_submitting_does_not_wait_for_the_work() ->   None  :
    """The whole reason a solve is a job. If submit blocked, the browser would
    get its job id when the sweep was already over."""
    k2=threading.Event()
    stuff2=JobRegistry()
    jobb= stuff2.submit(lambda send :k2.wait(timeout =5.0))
    assert stuff2.status(jobb.id)  in(JobStatus.PENDING, JobStatus.RUNNING)
    k2.set()
    stuff2.wait(jobb.id,timeout=5.0)

def test_work_that_raises_leaves_the_job_failed_with_the_reason() -> None :

    """phases/PHASE-7.md: a failed solve never reports a success it did not
    get, and it says what went wrong rather than spinning."""

    def explode(send :object)-> None :

        raise RuntimeError("the Jacobian is singular")
    Jobs  =  JobRegistry()


    jobb = Jobs.submit(explode)
    Jobs.wait(jobb.id, timeout =  5.0)
    assert Jobs.status(jobb.id)  is JobStatus.FAILED
    assert "singular" in Jobs.message(jobb.id)

def test_cancelling_stops_the_work_at_its_next_frame()->None:
    """Cancellation rides on the telemetry callback, which is the only place
    a solve looks up from the arithmetic. Sending a frame is what notices."""
    statred= threading.Event()
    FramesSent  =  [  ]

    def forever(send) ->None :
        statred.set ( )


        for n in range(1_000_000) :
            send(n)
            FramesSent.append(n)
    Jobs =  JobRegistry()
    filter =Jobs.submit(forever)
    statred.wait(timeout=5.0)


    assert  Jobs.cancel(filter.id )   is True
    Jobs.wait(filter.id,timeout=5.0)
    assert Jobs.status(filter.id)  is JobStatus.CANCELLED
    assert len(FramesSent)  < 1_000_000



def test_a_cancelled_job_ends_its_stream() -> None :

    """Otherwise a cancelled sweep leaves the browser waiting forever."""
    sta  =  threading.Event()
    def forever(send) -> None :


        sta.set ( )

        while True:
            send("tick")
    jbos  =   JobRegistry ( )
    str  = jbos.submit(forever)
    sta.wait(timeout = 5.0)
    jbos.cancel( str.id  )
    draiined =list(jbos.frames(str.id, timeout  = 5.0))



    assert draiined[- 1]== 'tick'
    assert jbos.status(str.id)is JobStatus.CANCELLED




def test_cancelling_a_finished_job_changes_nothing() -> None :
    """The browser cannot know the sweep ended a moment before the click."""
    Jobs =  JobRegistry (  )

    jobb =  Jobs.submit(lambda send :  send('one'))
    Jobs.wait(jobb.id,timeout= 5.0)
    assert Jobs.cancel(jobb.id) is False
    assert Jobs.status(jobb.id)is JobStatus.DONE


def test_the_work_can_see_that_it_was_cancelled(  )   ->  None  :
    """A solve holding a partial result should be able to put it down tidily
    rather than be stopped mid write. CancelledError is a real exception it can
    catch, not a thread kill."""

    Caught  :  list[  str ]  =  [ ]

    def tidy(send)-> None  :
        try :


            while  True :
                send('tick')
        except CancelledError :
            Caught.append('cleaned up')
            raise


    sta  = threading.Event()
    Jobs  = JobRegistry(); myvar  =   Jobs.submit ( lambda send  :  (  sta.set(  ),  tidy(send )  ))

    sta.wait(timeout =5.0)

    Jobs.cancel(myvar.id)
    Jobs.wait(myvar.id, timeout  = 5.0)

    assert Caught   ==  [ "cleaned up"]




def test_two_jobs_do_not_share_a_stream()->None:
    jbs=JobRegistry()


    fir  =  jbs.submit (  lambda send  :  send("first"))
    data2 =  jbs.submit(lambda send : send('second'))

    assert list(jbs.frames(fir.id,timeout =5.0)) ==["first"]
    assert list(jbs.frames(data2.id,timeout=5.0))==['second']


def test_an_unknown_job_is_refused()->None:
    Jobs= JobRegistry()

    with pytest.raises(KeyError) :
        Jobs.status("no-such-job")


def test_every_job_gets_its_own_id() -> None :
    Jobs=JobRegistry()
    idss={Jobs.submit(lambda send :None).id for _ in range(10)}


    assert len(  idss) ==   10
def test_a_full_queue_drops_its_oldest_frame_and_counts_the_loss() ->None:
    """phases/PHASE-7.md: the server drops frames rather than making the
    solver wait. Counting them is the difference between a plot that is
    coarse and a plot that is wrong without saying so."""
    jbos   =  JobRegistry( queue_size  =  4  )
    jobb=  jbos.submit(lambda send : [send(n) for n in range(10)])
    jbos.wait(jobb.id,timeout=5.0)
    assert  jbos.dropped( jobb.id  )   ==  6
    assert list(jbos.frames(jobb.id, timeout = 5.0))==  [6, 7, 8, 9]




def test_a_job_finishes_even_when_nobody_drains_its_queue()->None:
    """The queue fills exactly when no reader is attached, which is also when
    a blocking put would hang the worker forever. A job that never reaches a
    terminal state is a browser waiting on a result that will never come, and
    a process that cannot be shut down tidily."""

    Jobs  = JobRegistry( queue_size  = 2  )

    jobb=  Jobs.submit(lambda send:  [send(n)for n in range(50)])
    assert Jobs.wait(jobb.id,timeout =5.0) is JobStatus.DONE




def test_reading_frames_gives_up_rather_than_waiting_forever()-> None:
    """A websocket handler needs to be able to bound its wait. Without this
    a job whose worker is wedged holds a connection open indefinitely."""
    vals=JobRegistry()

    jobb = vals.submit(lambda send:threading.Event().wait(timeout= 5.0))

    with pytest.raises(TimeoutError):
        list(vals.frames(jobb.id,timeout = 0.05))




def test_waiting_for_a_job_gives_up_rather_than_hanging()->None:
    """Same argument on the other call, and it is the one the tests here
    lean on, so a silent hang would look like a slow suite."""
    joobs= JobRegistry()
    Job=joobs.submit(lambda send: threading.Event().wait(timeout = 5.0))

    with  pytest.raises ( TimeoutError )   :
        joobs.wait(Job.id,
             timeout= 0.05)


def test_a_finished_job_keeps_what_the_work_returned() -> None  :
    """A sweep's curve is the one thing a reader cannot afford to lose, and
    the frame queue is allowed to drop its oldest frame. So the return value
    is held on the job rather than sent, and a browser that connects after
    the last point still gets the curve."""
    max  = JobRegistry()
    Job =max.submit(lambda send :'the curve')

    max.wait(Job.id,timeout= 5.0)
    assert max.result(Job.id)=="the curve"



def test_a_job_that_is_still_running_has_no_result_yet() ->None :

    """None rather than a wait. The caller asks once the status is terminal,
    and a None that meant either not yet or nothing returned would be the
    same ambiguity a spinner is."""

    joobs = JobRegistry()
    hodling = threading.Event()
    Job=joobs.submit(lambda send:hodling.wait(timeout=5.0))
    try :
        assert joobs.result(Job.id) is None
    finally:
        hodling.set()
    joobs.wait ( Job.id ,   timeout  =   5.0 )


def test_a_failed_job_has_no_result() ->None :
    """Whatever the work built before it raised is not a result, and handing
    back a half filled curve as though it were one is the worst outcome
    available here."""
    jbos =  JobRegistry()

    Job=jbos.submit(lambda send :1/ 0)
    jbos.wait(Job.id,
        timeout  = 5.0)

    assert jbos.status (  Job.id)  is  JobStatus.FAILED
    assert jbos.result(Job.id) is None

def test_a_cancelled_job_keeps_nothing_either (  )   ->   None  :
    '''A cancelled sweep stopped somewhere nobody chose. Its partial curve is
    in the frames the reader already has.

    The work waits so that the cancel lands while it is still inside. Without
    that, this work finishes before the cancel arrives and the job is DONE
    with a result, which is the right answer to a different question.
    '''
    Jobs=JobRegistry()
    Cancelled = threading.Event()

    def work(send):
        send("one")
        Cancelled.wait(timeout  =5.0)
        send( "two" )
        return  "finished"
    junk= Jobs.submit(work);next(Jobs.frames(junk.id, timeout=  5.0))
    assert Jobs.cancel(junk.id)
    Cancelled.set()

    assert Jobs.wait(junk.id, timeout=  5.0) is JobStatus.CANCELLED
    assert Jobs.result(junk.id) is None


def  test_closing_the_registry_stops_every_running_job(  )  -> None :
    """A solver thread still running when the interpreter exits dies inside
    numpy, and Python reports exit code 120 over a test run that passed. So
    close cancels what is running and does not return until it has stopped."""
    jbs = JobRegistry()


    def  forever( send) ->  None  :
        while  True  :
            send("iteration")
    xx= [jbs.submit(forever)for _ in range(3)]
    Finished  =   jbs.submit(lambda  send   :   send( 'done'  )  ) ; jbs.wait(Finished.id,timeout= 5.0)

    jbs.close(timeout = 5.0)



    assert all(jbs.status(job.id) is JobStatus.CANCELLED for job in xx)

    assert jbs.status(Finished.id)is JobStatus.DONE



def test_closing_gives_up_on_work_that_never_reports()->None :
    """Cancellation lands at the next frame, so work that never sends one
    cannot be stopped. close says so rather than hanging the shutdown."""
    dir = threading.Event()
    joobs  = JobRegistry()
    joobs.submit(lambda send:dir.wait(timeout=5.0))

    with pytest.raises(TimeoutError)  :
        joobs.close(timeout =0.1)


    dir.set()



def test_a_full_registry_refuses_a_new_job_rather_than_queueing_it(  )  ->   None :
    """On a public server every submit is a thread spending CPU, so the number
    running at once is capped. The one over the cap is told the server is busy
    rather than slowing everyone else down."""
    vals  =  threading.Event(  )
    bb  = JobRegistry(max_running =2)
    hled =[bb.submit(lambda send :vals.wait(timeout= 5.0)) for _ in range(2)]


    with  pytest.raises(  BusyError  )  :
        bb.submit(lambda send : None)

    vals.set()
    for Job in hled  :
        bb.wait ( Job.id,  timeout   =   5.0  )
    bb.wait(bb.submit(lambda send : None).id, timeout=5.0)


def test_a_finished_job_is_forgotten_once_it_is_old_enough()->None:

    """Nothing else ever removes a job, so without this a public server
    keeps every curve anyone has solved until it restarts."""
    Now  =   [  0.0 ]
    joobs=  JobRegistry(keep_for =  60.0, clock =  lambda :  Now[0])
    Old = joobs.submit(lambda send  : None)
    joobs.wait(Old.id, timeout =  5.0)

    Now[0]  =  30.0
    joobs.wait(joobs.submit(lambda send : None).id, timeout= 5.0)
    assert joobs.status(Old.id)is JobStatus.DONE

    Now[0] = 61.0
    joobs.wait(joobs.submit(lambda send: None).id,timeout =5.0)

    with pytest.raises(KeyError) :
        joobs.status(  Old.id )




def  test_a_running_job_is_never_forgotten_however_old(  ) ->   None  :
    Now =   [0.0]
    relaese = threading.Event() ; d2 =JobRegistry(keep_for= 60.0,clock=lambda:Now[0])
    sow  =  d2.submit(lambda send  : relaese.wait(  timeout   =  5.0))



    Now[0] = 1000.0
    d2.wait(d2.submit(lambda send:None).id,timeout= 5.0)

    assert d2.status(sow.id)in(JobStatus.PENDING,JobStatus.RUNNING)
    relaese.set();  d2.wait(sow.id, timeout  = 5.0)

def test_a_job_past_its_time_limit_stops_and_says_why()  ->  None  :
    round =[0.0]
    foo = JobRegistry(time_limit=60.0, clock = lambda :round[0])



    def  forever (send)   -> None :
        while  True   :
            send("iteration") ; round[0] += 1.0

    Job = foo.submit(forever)


    assert foo.wait(Job.id, timeout=5.0) is JobStatus.CANCELLED
    assert "60 s" in foo.message(Job.id)
