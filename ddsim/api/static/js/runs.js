"use strict";

// The runs rail: one card per curve, each carrying a picture of itself.
//
// It used to be a line of text per run saying how that run's request differed
// from the next one's. The words are still there, because "Na 1e17" is what
// actually changed, but a shape beside them is what makes two runs comparable
// at a glance, which is the whole reason the plot keeps them.
//
// Nothing here computes a physical quantity. Every point was drawn by the
// solver and held as it arrived; all this does is map a value to a pixel, the
// same arithmetic drawCurve() does on the big canvas. The one logarithm is
// decades(), app.js's helper, which is a position on a screen.

const SPARK = { width: 100, height: 26 };
/** The viewBox a card's picture is drawn in. It is stretched to the card's
 * width, so these are proportions rather than pixels. */

// One frame over every run in the rail, so a curve carrying ten times the
// current of another sits visibly higher instead of being rescaled to look
// the same. Comparing them is the point.
function runSpan(runs, logY) {
  let lo = Infinity, hi = -Infinity, xlo = Infinity, xhi = -Infinity;
  for (const run of runs) {
    for (const point of run.points) {
      if (point.voltage < xlo) xlo = point.voltage;
      if (point.voltage > xhi) xhi = point.voltage;
      if (point.value === null || !isFinite(point.value)) continue;
      if (logY && !(point.value > 0)) continue;
      if (point.value < lo) lo = point.value;
      if (point.value > hi) hi = point.value;
    }
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo = logY ? lo / 10 : lo - 1; hi = logY ? hi * 10 : hi + 1; }
  if (!isFinite(xlo)) { xlo = 0; xhi = 1; }
  if (xlo === xhi) xhi = xlo + 1;
  return { lo: lo, hi: hi, xlo: xlo, xhi: xhi, logY: logY };
}

// The points of one run as an svg polyline, in the shared frame. A point the
// log axis cannot show breaks the line rather than being pinned to the floor,
// which is what the big plot does with it too.
function sparkline(points, span, colour) {
  const at = (point) => {
    const x = ((point.voltage - span.xlo) / (span.xhi - span.xlo)) * SPARK.width;
    const lo = span.logY ? decades(span.lo) : span.lo;
    const hi = span.logY ? decades(span.hi) : span.hi;
    const v = span.logY ? decades(point.value) : point.value;
    return x.toFixed(1) + "," + (SPARK.height - ((v - lo) / (hi - lo)) * SPARK.height).toFixed(1);
  };
  const drawable = points.filter(
    (point) =>
      point.value !== null &&
      isFinite(point.value) &&
      (!span.logY || point.value > 0)
  );
  if (!drawable.length) return "";
  // A sweep that stopped after its first point is still a result, so it gets
  // a dot where a line would have started.
  const mark = drawable.length === 1
    ? '<polyline points="' + at(drawable[0]) + " " + at(drawable[0]) +
      '" stroke-linecap="round" stroke-width="5"'
    : '<polyline points="' + drawable.map(at).join(" ") + '" stroke-width="1.4"';
  return (
    '<svg viewBox="0 0 ' + SPARK.width + " " + SPARK.height +
    '" preserveAspectRatio="none" aria-hidden="true">' + mark +
    ' fill="none" stroke="' + colour +
    '" vector-effect="non-scaling-stroke"></polyline></svg>'
  );
}

function runCard(label, points, span, colour, current) {
  const card = document.createElement("span");
  if (current) card.className = "current";
  const head = document.createElement("b");
  head.innerHTML = '<i class="swatch" style="background:' + colour + '"></i>';
  head.appendChild(document.createTextNode(label));
  card.appendChild(head);
  const picture = sparkline(points, span, colour);
  if (picture) card.insertAdjacentHTML("beforeend", picture);
  return card;
}

// The run on screen first, then every run kept under it, newest first, which
// is the order they were solved in reversed.
function showRuns() {
  const note = el("runs-note");
  note.textContent = "";
  const solved = state.points.length ? [{ points: state.points }] : [];
  const span = runSpan(state.runs.concat(solved), el("curve-log").checked);

  if (solved.length) {
    note.appendChild(runCard("this run", state.points, span, GREEN, true));
  }
  for (const run of state.runs.slice().reverse()) {
    note.appendChild(runCard(run.label, run.points, span, BLUE, false));
  }
}
