"""The DEVSIM side of the Phase 8 scoreboard.

Runs under `.venv-devsim`, like the other generators here, and writes into
`data/scoreboard/`. See phases/PHASE-8.md.

    .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/scoreboard.py api

`api` writes `devsim_api.txt`: every public name devsim exports, the solve
types and linear solvers its docstring lists as `solve:<type>` and
`solver:<type>`, the element types its Gmsh import accepts as
`create_gmsh_mesh:<element>`, and every function in its bundled
`python_packages` as `python_packages.<module>.<function>`. The capability
matrix cites these names, and test_scoreboard.py checks each citation against
this file, so a claim about what DEVSIM can do is a claim about this snapshot
rather than about my memory of its manual.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import os
import re
import sys

import devsim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data", "scoreboard")


def _choices(doc: str | None, argument: str) -> list[str]:
    """The {a, b, c} choices a devsim docstring lists for one argument."""
    match = re.search(rf"\b{argument} : \{{([^}}]*)\}}", doc or "")
    if match is None:
        raise RuntimeError(f"devsim's docstring no longer lists choices for {argument}")
    return [t.strip().strip("'") for t in match.group(1).split(",")]


def solve_types() -> list[str]:
    """solve's `type` and `solver_type` choices, e.g. solve:ac, solver:iterative."""
    doc = devsim.solve.__doc__
    return [f"solve:{t}" for t in _choices(doc, "type")] + [
        f"solver:{t}" for t in _choices(doc, "solver_type")
    ]


def gmsh_elements() -> list[str]:
    """The element types create_gmsh_mesh's docstring accepts, e.g. tetrahedron."""
    found = re.findall(r"- \d+ (\w+)", devsim.create_gmsh_mesh.__doc__ or "")
    if "triangle" not in found:
        raise RuntimeError("create_gmsh_mesh's docstring no longer lists element types")
    return [f"create_gmsh_mesh:{e}" for e in found]


def package_functions() -> list[str]:
    """Top level functions in devsim/python_packages, read from source."""
    import devsim.python_packages as packages

    folder = os.path.dirname(packages.__file__)
    names = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        module = filename[:-3]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                names.append(f"python_packages.{module}.{node.name}")
    return names


def write_api() -> str:
    public = sorted(n for n in dir(devsim) if not n.startswith("_"))
    today = datetime.date.today().isoformat()
    lines = [
        f"# devsim {devsim.__version__}",
        f"# written {today} by devsim_gen/scoreboard.py api",
        "# public names, solve types, gmsh elements, python_packages functions",
    ]
    lines += public
    lines += solve_types()
    lines += gmsh_elements()
    lines += package_functions()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "devsim_api.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["api"])
    args = parser.parse_args()
    if args.command == "api":
        print(write_api())
    return 0


if __name__ == "__main__":
    sys.exit(main())
