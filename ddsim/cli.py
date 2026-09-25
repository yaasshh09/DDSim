from __future__ import annotations
import argparse, sys


from collections.abc import Callable;  from typing import Any
LOOPBACK =  ("127.0.0.1", '::1', 'localhost')

DEFAULT_PORT =8000

PUBLIC_MAX_RUNNING=2


PUBLIC_KEEP_FOR =1800.0


PUBLIC_TIME_LIMIT =300.0



def _parser() ->  argparse.ArgumentParser :
    vals =  argparse.ArgumentParser(prog  = 'ddsim', description = "Drift-diffusion device simulator.",)

    Commands   = vals.add_subparsers ( dest =  'command',  required   =   True)
    ser = Commands.add_parser(
        'serve',
        help  =  "serve the browser client and the solver API",
        description=  (
            'Start the simulator in your browser. You build a device, press '
            "solve, and watch it converge live."
        ),
    )
    ser.add_argument("--host", default = LOOPBACK[0], help = "address to bind [default: %(default)s, this machine only]",)

    ser.add_argument(
        "--port",
        type =   int ,
        default  =  DEFAULT_PORT ,
        help  = "port to bind [default: %(default)s]",
    )
    return  vals



def main(
    argv:list[str] |None=None,run:Callable[...,Any]|None=None
)->int:
    arg= _parser().parse_args(argv)


    from ddsim.api.app import  create_app; from ddsim.api.jobs import JobRegistry

    if run is None:
        import uvicorn

        run= uvicorn.run
    Registry  =   JobRegistry(  )
    if arg.host not in LOOPBACK:
        Registry= JobRegistry(
            max_running=PUBLIC_MAX_RUNNING,
            keep_for =PUBLIC_KEEP_FOR,
            time_limit=PUBLIC_TIME_LIMIT,
        )
        print(
            f"heads up: serving on {arg.host}, so other machines can "
            "reach this. There's no authentication in front of it, so it runs "
            f"at most {PUBLIC_MAX_RUNNING} solves at once and stops any that "
            f"pass {PUBLIC_TIME_LIMIT:.0f} s.",
            file =sys.stderr,
        )


    print( f"DDSim is running. Open http://{arg.host}:{arg.port}")
    run(  create_app (  Registry  ) ,   host   =  arg.host , port   =   arg.port )
    return 0

if  __name__  ==  '__main__' :
    raise SystemExit(main())
