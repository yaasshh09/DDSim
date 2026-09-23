---
title: Bias continuation
summary: Walking the bias up in small steps from an answer already found, halving the step when one fails, and why the first point needs this too.
docs: 02-numerics.md#Bias continuation; 02-numerics.md#The first point is a jump too; 05-pitfalls.md#Specific traps; 07-decisions.md#Physics decisions log
---

## In plain words

Newton's method only converges if it starts close enough to the answer, and
no formula hands you a good guess for a diode at one volt forward bias or a
MOSFET with its drain already on. What you do have is the answer at a
slightly lower bias. So the simulator never jumps straight to the bias you
ask for. It solves at a bias it can reach, takes a small step, starts the
next solve from the answer it just found, and repeats until it gets there.
Each solve starts close to its own answer because the one before it was.

If a step is too big and the solve fails, nothing's lost. The step gets cut
in half and the solve tries again from the same good starting point. After a
success the step is allowed to grow back. If the step has been halved so many
times that it's tiny and the solve still fails, the sweep stops and tells you
where, since shrinking further won't help. The `step` knob is the biggest
bias step the solver takes between one recorded voltage and the next, and
`start` is where the sweep begins. On the residual plot, each dark dot marks
an attempt that got rejected and retried with a smaller step. The note beside
the plot says at what bias and what was done about it.

## In more depth

The driver is generic and knows nothing about semiconductors, so the same
code can do source stepping in the SPICE layer.

    value = start
    while value != target:
        try to solve at value + step, from the solution at value
        if it converged:  accept it, then step = min(1.5 * step, max_step)
        otherwise:        step = step / 2, and stop if step < min_step

The last step is clipped so it lands exactly on the target. A failed step is
halved from the size actually attempted, so a clipped attempt still retries
at a different point. Growth is 1.5, not 2, because doubling overshoots into
failure often enough to cost more than it saves. The floor defaults to a
thousandth of the first step, about ten halvings, and a ramp gives up after
200 attempts.

In a sweep, each requested voltage is the target of its own ramp, started
from the one before, and the step never grows past the `step` knob: 0.05 V by
default on the diode, 0.1 V on the transfer curve. So growth only wins back
size lost to a halving. A C-V sweep doesn't continue at all. Each of its
points is an independent equilibrium solve from a fresh guess, and the sweep
stops at the first bias that fails, returning the points before it.

The first point is a jump too. A transfer curve starts at zero gate with the
drain already at its bias, so the solve that begins the ramp is exactly the
jump the rule forbids. Measured on the 2835 node 1 um NMOS, cold from the
equilibrium guess: at 0 V applied, one Newton step with nothing clipped; at
0.25 V, ten steps with nothing clipped; at 0.5 V, twelve with two clipped;
at 1 V, twenty two with twelve clipped. Spending half a budget against the
step limit isn't a solve inside its basin. It's a solve that happens to
arrive, and rounding decides which side of that line a machine lands on. On
CI it landed on the wrong side and reported a residual of
$9.889 \times 10^{3}$ for the solve that converges locally. So the transfer
curve's cold first point gets ramped as well, on a fraction of every applied
bias at once, from the all zero device to the one asked for: first step 0.25
of the way, growing by 1.5. The final solve always happens at the full bias,
whether the ramp got there or not, so a stalled ramp can never hand back a
converged answer at a fraction nobody asked for. During that ramp the number
in a rejected note is the fraction, even though the note labels it in volts.
The diode sweep's first point is a Gummel solve from the equilibrium guess,
which on a $10^{16}$ cm$^{-3}$ diode can't be built above about 1.3 V, so a
high `start` there fails before any continuation begins.
