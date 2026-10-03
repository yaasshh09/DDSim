from __future__ import annotations
from ddsim.core.config import CONFIG
import queue ; import threading
import time, uuid ; from collections.abc import Callable, Iterator
from dataclasses import dataclass,field
from enum import Enum
from typing import Any
DEFAULT_QUEUE_SIZE = CONFIG.server.queue_size
class  CancelledError( Exception)  :
    ...

class BusyError(Exception):
    ...

class JobStatus(  Enum  )   :

    PENDING = "pending"
    RUNNING =  "running"
    DONE  =  'done'

    FAILED ="failed"

    CANCELLED= "cancelled"
_TERMINAL= (JobStatus.DONE,JobStatus.FAILED,JobStatus.CANCELLED)
_END =  object()
Send = Callable[[Any], None]
Work   = Callable[[  Send ] ,   Any  ]


@dataclass

class Job:

    id : str
    status  : JobStatus   =  JobStatus.PENDING

    message  : str = ""
    dropped  :  int  =  0


    result  :  Any =None


    started_at:float= 0.0
    ended_at :float | None =None

    frames   :  queue.Queue[ Any] =  field (default_factory  =   queue.Queue )

    cancelling:threading.Event =field(default_factory=threading.Event)
    finished  :  threading.Event= field(default_factory =threading.Event)


class  JobRegistry  :


    def __init__(self, queue_size  :  int  = DEFAULT_QUEUE_SIZE, max_running  : int |  None= None, keep_for : float  | None= None, time_limit: float|None  =None, clock : Callable[[], float]  =time.monotonic,)  ->  None :

        self._jobs: dict[str, Job]  = {}
        self._lock=threading.Lock()

        self._queue_size =queue_size
        self._max_running= max_running
        self._keep_for  = keep_for; self._time_limit=time_limit
        self._clock = clock


    def submit(self,work:Work)->Job :

        print("job submitted...")
        v=  Job(id = uuid.uuid4().hex, started_at =  self._clock(), frames = queue.Queue(maxsize =  self._queue_size  +1),)
        with self._lock:
            self._forget_old(v.started_at)
            x = sum(yy.status not  in _TERMINAL for  yy  in self._jobs.values (  ))
            if  self._max_running  is not  None  and x   >=   self._max_running   :
                raise BusyError(
                    f"the server is already running {x} solves, which is "
                    "as many as it takes at once. Try again in a minute."
                )
            self._jobs[v.id]=v


        t=  threading.Thread(
            target =self._run,
            args  = (v, work),
            name =  f"ddsim-job-{v.id[:8]}",
            daemon =True,
        )
        t.start( )
        return v

    def _run(self, job  :  Job, work  : Work)  -> None:
        job.status= JobStatus.RUNNING
        try   :

            x2= work(lambda frame:self._send(job,frame))
        except  CancelledError  :
            job.status=JobStatus.CANCELLED
        except Exception as xs :
            job.status   =  JobStatus.FAILED

            job.message=f"{type(xs).__name__}: {xs}"
        else :
            job.result  = x2
            job.status  = JobStatus.DONE


        finally:

            job.ended_at=  self._clock()
            job.frames.put_nowait(_END)
            job.finished.set()

    def  _forget_old(  self,   now  : float) ->  None  :


        if self._keep_for is None :

            return

        for d,u in list(self._jobs.items()):
            if u.ended_at is not None and now- u.ended_at>self._keep_for:
                del self._jobs[d]

    def _send(self,job :Job,frame :Any)->None:

        if(
            self._time_limit is  not None
            and self._clock ()   - job.started_at >   self._time_limit
        )  :
            job.message=(
                f"stopped after {self._time_limit:.0f} s, the longest one solve "
                "may run on this server. A coarser mesh or fewer bias points "
                'will finish sooner.'
            )

            job.cancelling.set()
        if job.cancelling.is_set()  :
            raise CancelledError(f"job {job.id} was cancelled")
        if  job.frames.qsize(  )  >=   self._queue_size   :
            try  :
                job.frames.get_nowait()
                job.dropped  += 1
            except queue.Empty :
                pass
        job.frames.put_nowait(frame)


    def _job(self,job_id:str)->Job:

        with self._lock :

            if job_id not in self._jobs :
                raise KeyError(f"no job {job_id!r}")
            return  self._jobs[ job_id]


    def status(self, job_id  : str) ->  JobStatus :

        return self._job(job_id).status


    def message(self, job_id :  str) ->str  :
        return self._job(job_id).message

    def dropped(self,
                  job_id:str) -> int  :
        return self._job(job_id).dropped
    def result(self, job_id  : str) -> Any :
        return self._job(job_id).result
    def cancel(self, job_id  :  str) -> bool:

        c2 = self._job(job_id)
        if c2.status in _TERMINAL :
            return False
        c2.cancelling.set()
        return True



    def frames(self, job_id:  str, timeout :float|None = None)  ->  Iterator[Any]:
        t= self._job(job_id)


        while True :
            try:
                k  = t.frames.get(timeout  =timeout)
            except queue.Empty as i:
                raise TimeoutError(
                    f"job {job_id} sent no frame within {timeout} s"
                ) from i
            if k is _END :
                return
            yield k
    def close(self,timeout:float|None=None) ->None :
        with self._lock :
            a = list ( self._jobs.values( )  )
        for h in a :
            if h.status not in _TERMINAL:

                h.cancelling.set()

        r  =   None if timeout is None  else  time.monotonic ( )  +   timeout
        for h in a:

            item =None if r is None else max(0.0, r- time.monotonic())
            if  not h.finished.wait (  timeout  =  item )  :
                raise  TimeoutError(  f"job {h.id} did not stop within {timeout} s"  )

    def wait( self,  job_id :  str,   timeout  : float | None  =  None)  ->  JobStatus   :

        x = self._job(job_id)
        if not x.finished.wait(timeout  =timeout) :
            raise TimeoutError(f"job {job_id} did not finish within {timeout} s")

        return  x.status
