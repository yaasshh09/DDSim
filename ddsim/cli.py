from __future__ import annotations
from ddsim.core.config import CONFIG
import argparse, sys
from collections.abc import Callable; from typing import Any
LOOPBACK =  ("127.0.0.1", '::1', 'localhost')

DEFAULT_PORT =CONFIG.server.port

PUBLIC_MAX_RUNNING=CONFIG.server.public_max_running


PUBLIC_KEEP_FOR =CONFIG.server.public_keep_for


PUBLIC_TIME_LIMIT =CONFIG.server.public_time_limit



def _parser() ->  argparse.ArgumentParser :
    val2 =  argparse.ArgumentParser(prog  = 'ddsim', description = "Drift-diffusion device simulator.",)

    t2   = val2.add_subparsers ( dest =  'command',  required   =   True)
    f = t2.add_parser(
        'serve',
        help  =  "serve the browser client and the solver API",
        description=  (
            'Start the simulator in your browser. You build a device, press '
            "solve, and watch it converge live."
        ),
    )
    f.add_argument("--host", default = LOOPBACK[0], help = "address to bind [default: %(default)s, this machine only]",)

    f.add_argument(
        "--port",
        type =   int ,
        default  =  DEFAULT_PORT ,
        help  = "port to bind [default: %(default)s]",
    )
    return  val2



def main(
    argv:list[str] |None=None,run:Callable[...,Any]|None=None
)->int:
    d= _parser().parse_args(argv)


    from ddsim.api.app import  create_app; from ddsim.api.jobs import JobRegistry

    if run is None:
        import uvicorn

        run= uvicorn.run
    e  =   JobRegistry(  )
    if d.host not in LOOPBACK:
        e= JobRegistry(
            max_running=PUBLIC_MAX_RUNNING,
            keep_for =PUBLIC_KEEP_FOR,
            time_limit=PUBLIC_TIME_LIMIT,
        )
        print(
            f"heads up: serving on {d.host}, so other machines can "
            "reach this. There's no authentication in front of it, so it runs "
            f"at most {PUBLIC_MAX_RUNNING} solves at once and stops any that "
            f"pass {PUBLIC_TIME_LIMIT:.0f} s.",
            file =sys.stderr,
        )


    print( f"DDSim is running. Open http://{d.host}:{d.port}")
    run(  create_app (  e  ) ,   host   =  d.host , port   =   d.port )
    return 0

if  __name__  ==  '__main__' :
    raise SystemExit(main())
