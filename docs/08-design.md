# The client's design system

What the page looks like and why. The single source of truth for every value
here is `ddsim/api/static/css/tokens.css`. This file is the reasoning; that
file is the definition. If the two disagree, the stylesheet is right and this
document is stale.

Nothing in here affects a number. The client computes no physical quantity,
and `tests/unit/test_client.py` holds it to that. What a design decision can
affect is whether a reader misreads a plot, which is why the three that could
are recorded as a dated row in `docs/07-decisions.md`.

## The idea

An instrument panel, not a web app. The solver is slow and honest and the
interface should be too: no motion that is not a state change, no surface that
is not holding something, and a number that never moves on the screen when
only its value changed.

## Layout

Three columns under a 46 px bar, each scrolling on its own so the knobs stay
reachable while a long plot column runs past the bottom of the screen.

| Region | Width | Holds |
|---|---|---|
| `#controls` | 312 px | device, sweep, models, the solve button |
| `#plots` | the rest | residual, curve, profile, and the lesson panel above them |
| `#rail` | 268 px | the runs kept on the plot, and the device file buttons |

The bar carries the wordmark, the phase, the status line and cancel. The dot
beside the status is lit exactly while a job runs, driven from
`#cancel:not(:disabled)` through `:has()` rather than from a second flag that
could disagree with the first.

## Type

Archivo for language, Roboto Mono for anything a number lands in. That split
is the point: a value under a live slider changes many times a second, and in
a proportional face every change reflows the row it sits in.

Both are vendored under `static/vendor/fonts` with their OFL text, because
`phases/PHASE-7.md` says the page works with no network.
`tools/vendor_client_libs.py` fetches them; `tests/unit/test_vendor.py` holds
every file to its hash and refuses any vendored stylesheet that still names a
remote host.

## Colour

Four surfaces, each separated by an explicit 1 px border rather than by a
shadow, so depth survives a dimmed screen. The ground is `#0b1417`, a deep
slate-teal and not black.

Six text tiers from `--ink` down to `--faint`. The design's two dimmest greys
measured 3.74:1 and 2.75:1 against their own backgrounds and were lifted to
4.80:1 and 3.17:1. `--faint` is held to the 3:1 bar for a user interface
component rather than the 4.5:1 bar for text, because it never draws text: it
is a disabled slider thumb and an idle status dot.

One accent and four trace colours. The traces are named by what they draw, and
the same four values appear twice: as `.swatch` rules in `css/panels.css` and
as the constants at the top of `js/app.js`, which is what the canvas is
actually painted with. Change one and change both.

| Quantity | Token | Value |
|---|---|---|
| psi, psi residual, Ec, the accent | `--signal` | `#5fd4d6` |
| n, n residual, Efn, a warning | `--warn` | `#e3a74f` |
| p, p residual, Efp | `--hot` | `#ea7a68` |
| gummel update, Ev | `--violet` | `#a992ef` |

The 2D field image uses a sequential ramp through the page ground, a mid teal
and the signal cyan. A rainbow was there before, and a rainbow invents a
visible boundary wherever its hue turns, which on a potential map reads as a
junction that is not there.

## What the constraints rule out

Tight radii, 3 to 6 px, and no pill except a run chip. Dense padding, 12 to
16 px in a panel. No gradient except the one that is a colour ramp carrying
data. No motion except the drawer's 0.12 s slide, which is the panel arriving
rather than an effect. Icons only where they carry a function: the explain
affordance is the letter `i` and not a glyph, because a letter still says what
it does at 10 px.

## The id contract

Every id on the page is a contract with `static/js` and with
`tests/unit/test_browser_smoke.py`, which drives 38 of them. Restyle them
freely. Rename none of them without changing both.

Two shapes are worth knowing before touching `css/controls.css`:

- A knob is built at runtime by `app.js` `knob()`: a label holding a name span
  and a `.control` span, with an optional range under the box. The explain
  button goes inside the name span there, but `learn.js` appends it to the
  label itself for anything carrying a `data-topic-id`. The label is a flex
  row ordered by role, so the control ends on the same right edge either way.
- The checkboxes are restyled, not replaced. Wrapping a hidden input in a
  painted label is the usual trick and it costs the element its box, which
  `test_browser_smoke.py` asserts on `#bands`. They keep their own 30 by 16
  box and paint themselves.
