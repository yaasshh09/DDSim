'''The command line. One command so far: `ddsim serve`.

phases/PHASE-7.md asks that someone who cloned the repo runs one command and
has a working page, with no build step that is not in the README. This is that
command, and the page it serves is a file in this repo.

The default binding is loopback. What is behind this port is an
unauthenticated process that will happily spend minutes of CPU on a request,
so putting it on a network is a decision rather than a default, and taking it
is noisy rather than silent.
'''


from __future__ import annotations
import argparse, sys


from collections.abc import Callable;  from typing import Any
LOOPBACK =  ("127.0.0.1", '::1', 'localhost')

"""Addresses that are only reachable from this machine."""

DEFAULT_PORT =8000

"""The port `ddsim serve` binds unless told otherwise."""

PUBLIC_MAX_RUNNING=2


"""Solves a registry reachable from other machines runs at once."""

PUBLIC_KEEP_FOR =1800.0


"""How long a finished solve stays readable on a public registry [s]."""
PUBLIC_TIME_LIMIT =300.0
"""Wall clock one solve may take on a public registry [s]. The default MOSFET
transfer sweep takes about 13 s on a laptop."""



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
    """Run one command.

    Args:
        argv: the arguments, or None to read them from the command line.
        run: the server to hand the application to. None means uvicorn, which
            is imported inside this function rather than at module scope so
            that importing this module does not require the serve extra.

    Returns a process exit code.
    """
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
