from __future__ import annotations
import threading
import pytest
from  ddsim.api.jobs import  BusyError, CancelledError,   JobRegistry,  JobStatus

def test_a_job_runs_and_finishes()-> None :
    val   =   JobRegistry(  )
    w= val.submit(lambda send:send('only frame'))
    val.wait( w.id, timeout  = 5.0 )

    assert  val.status(w.id )  is JobStatus.DONE



def  test_the_frames_arrive_in_the_order_they_were_sent(  )  -> None :
    j=JobRegistry()

    res2= j.submit(lambda send :[send(n)for n in range(5)])

    assert list(j.frames(res2.id,
                   timeout = 5.0))  == [0,
           1,
                   2,
      3,
                4]


def test_the_stream_ends_when_the_work_does() -> None:
    kk=JobRegistry()

    d=kk.submit(lambda send :send("done"))
    y =  list ( kk.frames( d.id,   timeout =  5.0 )  )

    assert y ==  ["done"]
    assert kk.status(d.id) is JobStatus.DONE



def  test_submitting_does_not_wait_for_the_work() ->   None  :
    j=threading.Event()
    item=JobRegistry()
    a2= item.submit(lambda send :j.wait(timeout =5.0))
    assert item.status(a2.id)  in(JobStatus.PENDING, JobStatus.RUNNING)
    j.set()
    item.wait(a2.id,timeout=5.0)

def test_work_that_raises_leaves_the_job_failed_with_the_reason() -> None :

    def explode(send :object)-> None :

        raise RuntimeError("the Jacobian is singular")
    c2  =  JobRegistry()


    c = c2.submit(explode)
    c2.wait(c.id, timeout =  5.0)
    assert c2.status(c.id)  is JobStatus.FAILED
    assert "singular" in c2.message(c.id)

def test_cancelling_stops_the_work_at_its_next_frame()->None:
    f= threading.Event()
    g  =  [  ]

    def forever(send) ->None :
        f.set ( )


        for n in range(1_000_000) :
            send(n)
            g.append(n)
    dat =  JobRegistry()
    bb =dat.submit(forever)
    f.wait(timeout=5.0)


    assert  dat.cancel(bb.id )   is True
    dat.wait(bb.id,timeout=5.0)
    assert dat.status(bb.id)  is JobStatus.CANCELLED
    assert len(g)  < 1_000_000



def test_a_cancelled_job_ends_its_stream() -> None :

    tmp  =  threading.Event()
    def forever(send) -> None :


        tmp.set ( )

        while True:
            send("tick")
    z  =   JobRegistry ( )
    nxt  = z.submit(forever)
    tmp.wait(timeout = 5.0)
    z.cancel( nxt.id  )
    e =list(z.frames(nxt.id, timeout  = 5.0))



    assert e[- 1]== 'tick'
    assert z.status(nxt.id)is JobStatus.CANCELLED




def test_cancelling_a_finished_job_changes_nothing() -> None :
    r =  JobRegistry (  )

    k =  r.submit(lambda send :  send('one'))
    r.wait(k.id,timeout= 5.0)
    assert r.cancel(k.id) is False
    assert r.status(k.id)is JobStatus.DONE


def test_the_work_can_see_that_it_was_cancelled(  )   ->  None  :

    dd  :  list[  str ]  =  [ ]

    def tidy(send)-> None  :
        try :


            while  True :
                send('tick')
        except CancelledError :
            dd.append('cleaned up')
            raise


    u  = threading.Event()
    x  = JobRegistry(); y  =   x.submit ( lambda send  :  (  u.set(  ),  tidy(send )  ))

    u.wait(timeout =5.0)

    x.cancel(y.id)
    x.wait(y.id, timeout  = 5.0)

    assert dd   ==  [ "cleaned up"]




def test_two_jobs_do_not_share_a_stream()->None:
    w2=JobRegistry()


    e  =  w2.submit (  lambda send  :  send("first"))
    d =  w2.submit(lambda send : send('second'))

    assert list(w2.frames(e.id,timeout =5.0)) ==["first"]
    assert list(w2.frames(d.id,timeout=5.0))==['second']


def test_an_unknown_job_is_refused()->None:
    j= JobRegistry()

    with pytest.raises(KeyError) :
        j.status("no-such-job")


def test_every_job_gets_its_own_id() -> None :
    c=JobRegistry()
    it={c.submit(lambda send :None).id for _ in range(10)}


    assert len(  it) ==   10
def test_a_full_queue_drops_its_oldest_frame_and_counts_the_loss() ->None:
    thing   =  JobRegistry( queue_size  =  4  )
    f=  thing.submit(lambda send : [send(n) for n in range(10)])
    thing.wait(f.id,timeout=5.0)
    assert  thing.dropped( f.id  )   ==  6
    assert list(thing.frames(f.id, timeout = 5.0))==  [6, 7, 8, 9]




def test_a_job_finishes_even_when_nobody_drains_its_queue()->None:

    xs  = JobRegistry( queue_size  = 2  )

    s2=  xs.submit(lambda send:  [send(n)for n in range(50)])
    assert xs.wait(s2.id,timeout =5.0) is JobStatus.DONE




def test_reading_frames_gives_up_rather_than_waiting_forever()-> None:
    x2=JobRegistry()

    j = x2.submit(lambda send:threading.Event().wait(timeout= 5.0))

    with pytest.raises(TimeoutError):
        list(x2.frames(j.id,timeout = 0.05))




def test_waiting_for_a_job_gives_up_rather_than_hanging()->None:
    y= JobRegistry()
    x=y.submit(lambda send: threading.Event().wait(timeout = 5.0))

    with  pytest.raises ( TimeoutError )   :
        y.wait(x.id,
             timeout= 0.05)


def test_a_finished_job_keeps_what_the_work_returned() -> None  :
    f  = JobRegistry()
    u =f.submit(lambda send :'the curve')

    f.wait(u.id,timeout= 5.0)
    assert f.result(u.id)=="the curve"



def test_a_job_that_is_still_running_has_no_result_yet() ->None :

    w = JobRegistry()
    prev = threading.Event()
    i=w.submit(lambda send:prev.wait(timeout=5.0))
    try :
        assert w.result(i.id) is None
    finally:
        prev.set()
    w.wait ( i.id ,   timeout  =   5.0 )


def test_a_failed_job_has_no_result() ->None :
    el =  JobRegistry()

    buf=el.submit(lambda send :1/ 0)
    el.wait(buf.id,
        timeout  = 5.0)

    assert el.status (  buf.id)  is  JobStatus.FAILED
    assert el.result(buf.id) is None

def test_a_cancelled_job_keeps_nothing_either (  )   ->   None  :
    dd=JobRegistry()
    r = threading.Event()

    def work(send):
        send("one")
        r.wait(timeout  =5.0)
        send( "two" )
        return  "finished"
    ss= dd.submit(work);next(dd.frames(ss.id, timeout=  5.0))
    assert dd.cancel(ss.id)
    r.set()

    assert dd.wait(ss.id, timeout=  5.0) is JobStatus.CANCELLED
    assert dd.result(ss.id) is None


def  test_closing_the_registry_stops_every_running_job(  )  -> None :
    flag = JobRegistry()


    def  forever( send) ->  None  :
        while  True  :
            send("iteration")
    c= [flag.submit(forever)for _ in range(3)]
    buf  =   flag.submit(lambda  send   :   send( 'done'  )  ) ; flag.wait(buf.id,timeout= 5.0)

    flag.close(timeout = 5.0)



    assert all(flag.status(jj.id) is JobStatus.CANCELLED for jj in c)

    assert flag.status(buf.id)is JobStatus.DONE



def test_closing_gives_up_on_work_that_never_reports()->None :
    thing = threading.Event()
    y  = JobRegistry()
    y.submit(lambda send:thing.wait(timeout=5.0))

    with pytest.raises(TimeoutError)  :
        y.close(timeout =0.1)


    thing.set()



def test_a_full_registry_refuses_a_new_job_rather_than_queueing_it(  )  ->   None :
    x  =  threading.Event(  )
    t  = JobRegistry(max_running =2)
    ok =[t.submit(lambda send :x.wait(timeout= 5.0)) for _ in range(2)]


    with  pytest.raises(  BusyError  )  :
        t.submit(lambda send : None)

    x.set()
    for c in ok  :
        t.wait ( c.id,  timeout   =   5.0  )
    t.wait(t.submit(lambda send : None).id, timeout=5.0)


def test_a_finished_job_is_forgotten_once_it_is_old_enough()->None:

    obj  =   [  0.0 ]
    j=  JobRegistry(keep_for =  60.0, clock =  lambda :  obj[0])
    tt = j.submit(lambda send  : None)
    j.wait(tt.id, timeout =  5.0)

    obj[0]  =  30.0
    j.wait(j.submit(lambda send : None).id, timeout= 5.0)
    assert j.status(tt.id)is JobStatus.DONE

    obj[0] = 61.0
    j.wait(j.submit(lambda send: None).id,timeout =5.0)

    with pytest.raises(KeyError) :
        j.status(  tt.id )




def  test_a_running_job_is_never_forgotten_however_old(  ) ->   None  :
    w =   [0.0]
    k = threading.Event() ; w2 =JobRegistry(keep_for= 60.0,clock=lambda:w[0])
    vals  =  w2.submit(lambda send  : k.wait(  timeout   =  5.0))



    w[0] = 1000.0
    w2.wait(w2.submit(lambda send:None).id,timeout= 5.0)

    assert w2.status(vals.id)in(JobStatus.PENDING,JobStatus.RUNNING)
    k.set();  w2.wait(vals.id, timeout  = 5.0)

def test_a_job_past_its_time_limit_stops_and_says_why()  ->  None  :
    v =[0.0]
    w = JobRegistry(time_limit=60.0, clock = lambda :v[0])



    def  forever (send)   -> None :
        while  True   :
            send("iteration") ; v[0] += 1.0

    b = w.submit(forever)


    assert w.wait(b.id, timeout=5.0) is JobStatus.CANCELLED
    assert "60 s" in w.message(b.id)
