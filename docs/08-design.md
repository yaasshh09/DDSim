# The client's design system

What the page looks like, and why. The single source of truth for every value
here is `ddsim/api/static/css/tokens.css`. This file is the reasoning and that
one is the definition. If they disagree, the stylesheet is right and this
document is out of date.

Nothing in here affects a number. The client computes no physical quantity,
and `tests/unit/test_client.py` holds it to that. What a design decision can
affect is whether someone misreads a plot, which is why the three decisions
that could are recorded as a dated row in `docs/07-decisions.md`.

## The idea

An instrument panel, not a web app. The solver is slow and honest, and the
interface should be too: no motion that isn't a change of state, no surface
that isn't holding something, and no number that jumps around on screen when
only its value changed.

## Layout

Three columns under a 46 px bar, each scrolling on its own, so the knobs stay
in reach while a long column of plots runs off the bottom of the screen.

| Region | Width | Holds |
|---|---|---|
| `#controls` | 312 px | device, sweep, models, the solve button |
| `#plots` | the rest | residual, curve, profile, and the lesson panel above them |
| `#rail` | 268 px | the runs kept on the plot, and the device file buttons |

The bar carries the wordmark, the Build and Results tabs, the "what is this?"
button, the status line and cancel. The dot beside the status is lit exactly
while a job runs. It's driven from `#cancel:not(:disabled)` through `:has()`
rather than from a second flag that could disagree with the first.

## Type

Poppins, in two weights. 500 carries everything that gets read, and 600
everything that's a heading or a value worth finding, so the page has one axis
of emphasis instead of four.

**Poppins has no tabular figures, and its digits are proportional.** A one is
350 units wide against a zero's 647, and there's no `tnum` feature to switch
on, so `font-variant-numeric` does nothing here. That matters because a value
under a live slider gets rewritten many times a second. It's handled with
geometry instead of the font: every number the page shows sits in its own
fixed width box. `#state`, which is loose text in the header and rewrites
several numbers a second during a solve, sits after the header's flex gap, so
the cancel button stays pinned to the right edge and the changing width only
moves the gap. A `min-width` there once left `ready` floating a long way from
cancel. If a number ever has to sit in free flowing text and stay still, it
needs a box or a pinned width too.

The family is vendored under `static/vendor/fonts` with its OFL text, because
`phases/PHASE-7.md` says the page has to work with no network. It isn't
downloaded at build time: the release goes in `fonts/`, which is gitignored,
and `tools/vendor_client_libs.py` subsets the two weights down to the
characters this page draws. That subsetting is most of the point. Poppins
ships Devanagari, which this page never uses and which is nine tenths of the
file, so each weight drops from 156 KB to under 17 KB and the vendored tree is
33 KB of font.

`tests/unit/test_vendor.py` holds every file to its hash and refuses any
vendored stylesheet that still names a remote host.

## Colour

Four surfaces, each separated by an explicit 1 px border instead of a shadow,
so the depth survives a dimmed screen. The ground is `#0b1417`, a deep slate
teal, not black.

Six text tiers run from `--ink` down to `--faint`. The design's two dimmest
greys measured 3.74:1 and 2.75:1 against their own backgrounds, and I lifted
them to 4.80:1 and 3.17:1. `--faint` is held to the 3:1 bar for a user
interface component rather than the 4.5:1 bar for text, because it never draws
text: it's a disabled slider thumb and an idle status dot.

One accent and four trace colours. The traces are named for what they draw,
and the same four values appear in two places: as `.swatch` rules in
`css/panels.css`, and as the constants at the top of `js/app.js`, which is what
the canvas actually gets painted with. Change one, change both.

| Quantity | Token | Value |
|---|---|---|
| psi, psi residual, Ec, the accent | `--signal` | `#5fd4d6` |
| n, n residual, Efn, a warning | `--warn` | `#e3a74f` |
| p, p residual, Efp | `--hot` | `#ea7a68` |
| gummel update, Ev | `--violet` | `#a992ef` |

The 2D field image uses a sequential ramp through the page ground, a mid teal
and the signal cyan. It used to be a rainbow, and a rainbow invents a visible
boundary wherever its hue turns, which on a potential map reads as a junction
that isn't there.

## What the constraints rule out

Tight radii, 3 to 6 px, and no pills except a run chip. Dense padding, 12 to
16 px in a panel. No gradients except the one that's a colour ramp carrying
data. No motion except the drawer's 0.12 s slide, which is the panel arriving,
not an effect. Icons only where they do something: the explain button is the
letter `i` rather than a glyph, because a letter still says what it does at
the size a control label sits at.

## Reading for two audiences

Every knob has a plain name and its argument name. `KNOB_LABELS` in
`ddsim/api/learn.py` holds the words, keyed by argument name next to
`KNOB_TOPICS`, for the same reason: a name two devices share means the same
thing on both. The schema serves it as `label`, and the form leads with it and
sets the symbol underneath in mono. A knob missing from the map falls back to
its own argument name, so a knob added to a constructor still shows up.

Nothing under 11 px, no exceptions, the symbol line included. The design's
own ramp annotated at 9 to 10.5 px, and that didn't survive review: a label, a
legend key, a details summary and a panel button are functional text whatever
the ramp says.

## The two halves of the stage

`#stage` holds `#plots` and `#build`, and exactly one of them is on screen.
The rails never move, because setting a device's mesh and drawing its shape
are the same job, and the solve button belongs to both.

Picking a drawn device turns the stage to Build. A solve that finishes turns
it to Results, and a refused one doesn't, because the refusal names the part
that was wrong, and the editor has to stay open for that to be worth reading.

The editor is split three ways. `builder.js` owns the frame: the groups, the
words above them, and the column headings that turn a row of bare boxes into
a table. `preview.js` owns the canvas. `drawing.js` owns the rows and the
drag. None of them is over 250 lines, which is why they're three files.

## The id contract

Every id on the page is a contract with `static/js` and with
`tests/unit/test_browser_smoke.py`, which drives 38 of them. Restyle them
freely. Don't rename one without changing both.

Two shapes are worth knowing before you touch `css/controls.css`:

- A knob is built at runtime by `app.js` `knob()`: a label holding a name span
  and a `.control` span, with an optional range under the box. The explain
  button goes inside the name span there, but `learn.js` appends it to the
  label itself for anything carrying a `data-topic-id`. The label is a flex
  row ordered by role, so the control ends on the same right edge either way.
- The checkboxes are restyled, not replaced. Wrapping a hidden input in a
  painted label is the usual trick, and it costs the element its box, which
  `test_browser_smoke.py` asserts on for `#bands`. They keep their own 30 by
  16 box and paint themselves.
