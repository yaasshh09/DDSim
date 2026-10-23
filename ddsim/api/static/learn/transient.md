---
title: The transient sweep
summary: A voltage step on one contact, followed in time, with a resistor in series and a step size the solver picks itself.
docs: numerics.md#Time integration (Phase 11)
---

## In plain words

Every other sweep asks what the device settles to. This one asks how it gets
there. The contact sits at the first voltage long enough to be steady, then
the source jumps to the second voltage and I follow what happens afterwards.
The times box lists the moments to keep a snapshot at, counted from the jump,
in seconds. The solver takes as many steps as it needs between them.

The series resistor sits between the source and the contact. At zero the
source drives the contact directly, so the contact voltage jumps right away.
With a resistor, the current through it sets how fast the contact voltage can
move. That's how a diode's reverse recovery shows up: switch a forward biased
diode off through a resistor and the stored holes keep the junction forward
for a while, with a steady reverse current, before it finally lets go.

Time step accuracy is how much error each step may make. Smaller is more
accurate and takes more steps. The curve plots current against the log of
time, and the play button walks the profile through the snapshots.

## In more depth

Each step is BDF2 with variable steps. With $\Delta t$ the step being taken
and $r$ its ratio to the step before,

$$\frac{M}{\Delta t}\left(\frac{1+2r}{1+r}\,x_{k+1} - (1+r)\,x_k + \frac{r^2}{1+r}\,x_{k-1}\right) + F(x_{k+1}) = 0$$

solved by Newton. The first step has no history, so it's backward Euler. The
step error is estimated from the gap between the quadratic predictor through
the last three states and the converged corrector, scaled per unknown by
`rtol`, with a floor of `rtol` V on $\psi$ and `rtol` $n_i$ on the densities.
A step whose error passes 1 is thrown away and retried smaller, and the next
step is resized by $0.9\,\mathrm{err}^{-1/3}$, held between 0.2 and 2 times
the last one. A step Newton can't converge is halved, up to 10 times.

With a resistor, each step solves for the contact voltage $V$ that makes the
device current equal $(V_{source} - V)/R$, by bracketing and then brentq. The
terminal current is conduction plus displacement, the time derivative of the
contact charge, taken with the same BDF2 weights as the step. Without that
term the terminal currents stop summing to zero during a transient.

`resistance` is in ohm cm^2 on the 1D diode, since current there is per area,
and ohm cm on a 2D device, where current is per width.
