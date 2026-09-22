"use strict";

// The start screen, phases/PHASE-7.md. Four things worth solving, said in
// words somebody who has never met a MOSFET can follow, each one setting the
// form up and getting out of the way.
//
// It is a panel at the top of the stage rather than a modal over it. A modal
// would swallow the first click on everything underneath, which costs the
// browser tests their first action and buys nothing: the plots behind it are
// empty until something has been solved anyway.
//
// Nothing here computes. Each card names a device kind and a sweep kind that
// the schema already carries, and setUp() puts them on the form.

// A small schematic of each device, drawn from the same palette as the page.
// Not decoration: the shape is what the words are describing, and seeing the
// gate sit on top of the oxide is most of the explanation.
const CARD_ART = {
  pn_diode:
    '<rect x="4" y="14" width="46" height="32" fill="#2d4a52"></rect>' +
    '<rect x="50" y="14" width="46" height="32" fill="#3d616a"></rect>' +
    '<line x1="50" y1="10" x2="50" y2="50" stroke="#e3a74f" stroke-dasharray="3 3"></line>' +
    '<text x="18" y="34" fill="#e8f1f2" font-size="11">p</text>' +
    '<text x="72" y="34" fill="#e8f1f2" font-size="11">n</text>',
  mos_cap:
    '<rect x="8" y="12" width="84" height="7" fill="#a992ef"></rect>' +
    '<rect x="8" y="19" width="84" height="8" fill="#16252a"></rect>' +
    '<rect x="8" y="27" width="84" height="21" fill="#2d4a52"></rect>' +
    '<text x="38" y="18" fill="#0b1417" font-size="8">gate</text>' +
    '<text x="34" y="42" fill="#e8f1f2" font-size="9">silicon</text>',
  nmos:
    '<rect x="4" y="22" width="92" height="26" fill="#2d4a52"></rect>' +
    '<rect x="4" y="22" width="22" height="14" fill="#5fd4d6"></rect>' +
    '<rect x="74" y="22" width="22" height="14" fill="#5fd4d6"></rect>' +
    '<rect x="30" y="18" width="40" height="5" fill="#16252a"></rect>' +
    '<rect x="30" y="12" width="40" height="6" fill="#a992ef"></rect>' +
    '<text x="6" y="33" fill="#0b1417" font-size="8">src</text>' +
    '<text x="76" y="33" fill="#0b1417" font-size="8">drn</text>',
  drawing:
    '<rect x="6" y="12" width="88" height="36" fill="none" stroke="#3d616a" ' +
    'stroke-dasharray="4 3"></rect>' +
    '<rect x="18" y="22" width="34" height="18" fill="#2d4a52"></rect>' +
    '<rect x="52" y="28" width="26" height="12" fill="#16252a"></rect>' +
    '<path d="M62 18 l0 12 l3.5 -3.5 l2.5 5 l2.5 -1.2 l-2.5 -5 l4.5 -0.8 z" ' +
    'fill="#e8f1f2"></path>',
};

// kind, sweep, and what the thing actually is. The blurb answers "why would
// I press this", not "what is it called".
const CARDS = [
  {
    kind: "pn_diode",
    sweep: "iv",
    title: "A diode",
    blurb:
      "Two pieces of silicon doped differently, touching. It passes current " +
      "one way and blocks it the other, and nothing in the solver was told " +
      "to do that.",
    shows: "Current against voltage · about a second",
  },
  {
    kind: "mos_cap",
    sweep: "cv",
    title: "A MOS capacitor",
    blurb:
      "Metal over oxide over silicon. Put a voltage on the metal and the " +
      "silicon underneath empties of carriers, then fills with the opposite " +
      "kind. This is the inside of every transistor.",
    shows: "Capacitance against voltage · a few seconds",
  },
  {
    kind: "nmos",
    sweep: "transfer",
    title: "A transistor",
    blurb:
      "A switch with no moving parts. Voltage on the gate opens a channel " +
      "between source and drain, and the current climbs by decades over a " +
      "fraction of a volt.",
    shows: "Drain current against gate voltage · under a minute",
  },
  {
    kind: "drawing",
    sweep: "transfer",
    title: "Draw your own",
    blurb:
      "Rectangles of silicon, oxide and doping, with contacts on them. It " +
      "starts as the transistor above so there is something to take apart.",
    shows: "Opens the editor · solve when you are ready",
  },
];

const LEDE =
  "This solves the equations that describe how electrons and holes move " +
  "through a piece of silicon, then draws what it found. Nothing here is " +
  "fitted or faked: shapes and doping go in, current comes out. Pick a " +
  "device to start with, or draw one yourself.";

const MORE =
  "Rather be walked through it? Pick a guided lesson from the top of the " +
  "left column. Every number, plot and switch on this page has an " +
  "<b>i</b> beside it that explains what it is.";

const SEEN = "ddsim.welcome.seen";

function showWelcome(show) {
  el("welcome").hidden = !show;
}

function buildWelcome() {
  el("welcome-lede").textContent = LEDE;
  el("welcome-more").innerHTML = MORE;
  const into = el("welcome-cards");
  into.textContent = "";
  for (const card of CARDS) {
    if (!state.schema.devices[card.kind]) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "start";
    button.dataset.kind = card.kind;
    button.innerHTML =
      '<svg viewBox="0 0 100 60" aria-hidden="true">' + CARD_ART[card.kind] +
      "</svg><b>" + card.title + "</b><span>" + card.blurb +
      '</span><i class="shows">' + card.shows + "</i>";
    button.addEventListener("click", () => startFrom(card));
    into.appendChild(button);
  }
}

// Put one card's device and sweep on the form, take the coarse mesh where the
// device offers one, and leave. Nothing solves: pressing solve stays the
// student's decision, because a solve takes real time.
function startFrom(card) {
  el("device-kind").value = card.kind;
  onDeviceKind();
  if (state.schema.sweeps[card.sweep]) {
    el("sweep-kind").value = card.sweep;
    onSweepKind();
  }
  if (state.schema.presets[card.kind]) useMesh(true);
  dismissWelcome();
}

function dismissWelcome() {
  showWelcome(false);
  try {
    localStorage.setItem(SEEN, "1");
  } catch (blocked) {
    // A private window refuses storage. The start screen simply comes back
    // next time, which is a smaller problem than a page that will not load.
  }
}

function openWelcome() {
  buildWelcome();
  showWelcome(true);
  el("welcome").scrollIntoView({ block: "start" });
}

function welcome() {
  let seen = false;
  try {
    seen = localStorage.getItem(SEEN) === "1";
  } catch (blocked) {
    seen = false;
  }
  buildWelcome();
  showWelcome(!seen);
}

el("welcome-skip").addEventListener("click", dismissWelcome);
el("welcome-open").addEventListener("click", openWelcome);
