"""HTTP and WebSocket access to the solver, phases/PHASE-7.md.

This package may only call the functions the CLI would call. It holds no
physics of its own and it adds no branch that changes a solved number. The
import graph test in tests/unit/test_linear.py enforces the direction: this
package imports from device/, extract/ and solve/, and nothing imports it.
"""
