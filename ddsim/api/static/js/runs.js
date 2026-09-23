"use strict";

const SPARK = { width: 100, height: 26 };

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
