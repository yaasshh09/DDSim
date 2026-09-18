"use strict";

const BLUE = "#1f6feb", PURPLE = "#8250df", GREEN = "#1a7f37", RED = "#b23c17";

const el = (id) => document.getElementById(id);

// How long the page waits after a slider stops moving before it submits.
// Shorter than a 1D solve so a drag feels live, and long enough that one
// sweep of the hand is one job rather than five.
const LIVE_DELAY = 200;

const state = {
  schema: null,
  job: null,
  socket: null,
  residual: [],   // {value, colour, families}, one per iteration or cycle
  newton: null,   // the last Newton split, {residual, update} by family
  stalled: null,  // that split as it was when a step was last rejected
  rejected: [],   // positions in residual where a step was rejected
  points: [],     // {voltage, value}
  curve: null,
  fields: null,
  cutline: null,  // {from, to} in fractional mesh indices, on a 2D image
  valueName: "current",
  runs: [],       // finished runs kept as overlays: {points, request, label}
  request: null,  // what the run on screen was asked for
  pending: null,  // the timer a moving slider keeps resetting
};

// ---------------------------------------------------------------- plotting

function fit(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = Number(canvas.getAttribute("height"));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const pen = canvas.getContext("2d");
  pen.setTransform(ratio, 0, 0, ratio, 0, 0);
  pen.clearRect(0, 0, width, height);
  return { pen: pen, width: width, height: height };
}

function ink(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// A decade is a position on a screen. Nothing physical is computed from it.
const decades = (v) => Math.log10(v);
const undecades = (v) => Math.pow(10, v);

function format(v) {
  if (v === 0) return "0";
  const size = Math.abs(v);
  if (size >= 1e4 || size < 1e-3) return v.toExponential(1);
  return String(Math.round(v * 1000) / 1000);
}

function axes(pen, box, xs, ys, options) {
  const logY = Boolean(options && options.logY);
  // Two frames sharing one box leave room on the right for the second one.
  const twin = Boolean(options && options.twin);
  const right = Boolean(options && options.right);
  const pad = { left: 62, right: twin ? 62 : 12, top: 10, bottom: 22 };
  const inner = {
    x: pad.left,
    y: pad.top,
    width: Math.max(10, box.width - pad.left - pad.right),
    height: Math.max(10, box.height - pad.top - pad.bottom),
  };

  let lo = Infinity, hi = -Infinity;
  for (const y of ys) {
    if (y === null || !isFinite(y)) continue;
    if (logY && !(y > 0)) continue;
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo = logY ? lo / 10 : lo - 1; hi = logY ? hi * 10 : hi + 1; }

  let xlo = Infinity, xhi = -Infinity;
  for (const x of xs) {
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
  }
  if (!isFinite(xlo)) { xlo = 0; xhi = 1; }
  if (xlo === xhi) xhi = xlo + 1;

  const yOf = (y) => {
    if (logY) {
      const a = decades(lo), b = decades(hi);
      return inner.y + inner.height - ((decades(y) - a) / (b - a)) * inner.height;
    }
    return inner.y + inner.height - ((y - lo) / (hi - lo)) * inner.height;
  };
  const xOf = (x) => inner.x + ((x - xlo) / (xhi - xlo)) * inner.width;

  pen.strokeStyle = ink("--grid");
  pen.lineWidth = 1;
  pen.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = inner.y + (i / 4) * inner.height;
    pen.moveTo(inner.x, y);
    pen.lineTo(inner.x + inner.width, y);
  }
  pen.stroke();

  pen.fillStyle = (options && options.ink) || ink("--muted");
  pen.font = "11px ui-monospace, monospace";
  pen.textAlign = right ? "left" : "right";
  const labelX = right ? inner.x + inner.width + 6 : inner.x - 6;
  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    const value = logY
      ? undecades(decades(hi) - t * (decades(hi) - decades(lo)))
      : hi - t * (hi - lo);
    pen.fillText(format(value), labelX, inner.y + t * inner.height + 4);
  }
  pen.fillStyle = ink("--muted");
  pen.textAlign = "left";
  pen.fillText(format(xlo), inner.x, box.height - 6);
  pen.textAlign = "right";
  pen.fillText(format(xhi), inner.x + inner.width, box.height - 6);

  return { xOf: xOf, yOf: yOf, inner: inner, logY: logY };
}

function line(pen, frame, xs, ys, colour, options) {
  const usable = (y) => y !== null && isFinite(y) && !(frame.logY && !(y > 0));
  pen.strokeStyle = colour;
  pen.lineWidth = 1.5;
  pen.beginPath();
  let down = false;
  for (let i = 0; i < xs.length; i++) {
    if (!usable(ys[i])) { down = false; continue; }
    const px = frame.xOf(xs[i]), py = frame.yOf(ys[i]);
    if (down) pen.lineTo(px, py); else pen.moveTo(px, py);
    down = true;
  }
  pen.stroke();

  if (options && options.dots) {
    pen.fillStyle = colour;
    for (let i = 0; i < xs.length; i++) {
      if (!usable(ys[i])) continue;
      pen.beginPath();
      pen.arc(frame.xOf(xs[i]), frame.yOf(ys[i]), 2.5, 0, 6.2832);
      pen.fill();
    }
  }
}

// ------------------------------------------------------------- three plots

function drawResidual() {
  const box = fit(el("residual"));
  if (!state.residual.length) return;
  const logY = el("residual-log").checked;
  const xs = state.residual.map((_, i) => i);
  // The axis spans every family, not just the largest, or the smaller two
  // would fall off the bottom of a log plot.
  const ys = state.residual.flatMap((r) =>
    r.families ? Object.values(r.families).concat([r.value]) : [r.value]
  );
  const frame = axes(box.pen, box, xs, ys, { logY: logY });

  // A Newton iteration that came with its split draws one line per equation
  // family, so a stall shows which equation is stuck. One without draws its
  // single residual, and a Gummel cycle its update.
  const split = (family) => (r) => (r.families ? r.families[family] : null);
  const series = [
    [BLUE, (r) => (r.colour === BLUE ? (r.families ? r.families.psi : r.value) : null)],
    [GREEN, split("n")],
    [RED, split("p")],
    [PURPLE, (r) => (r.colour === PURPLE ? r.value : null)],
  ];
  for (const [colour, pick] of series) {
    line(box.pen, frame, xs, state.residual.map(pick), colour);
  }
  box.pen.fillStyle = ink("--ink");
  for (const at of state.rejected) {
    const r = state.residual[at];
    if (!r || !isFinite(r.value) || (logY && !(r.value > 0))) continue;
    box.pen.beginPath();
    box.pen.arc(frame.xOf(at), frame.yOf(r.value), 3.5, 0, 6.2832);
    box.pen.fill();
  }
}

// Every run that finished is still on the plot, faded, until it is cleared.
// An overlay is the points that run was drawn with, held as they were: no
// overlay is ever solved again, so the earlier curve cannot drift when the
// knobs move under it.
function drawCurve() {
  const box = fit(el("curve"));
  if (!state.points.length && !state.runs.length) return;

  // One frame over every run on the plot, or the older ones would be drawn
  // against an axis that does not reach them.
  const xs = [], ys = [];
  for (const run of state.runs.concat([{ points: state.points }])) {
    for (const point of run.points) {
      xs.push(point.voltage);
      ys.push(point.value);
    }
  }
  const frame = axes(box.pen, box, xs, ys, { logY: el("curve-log").checked });

  box.pen.globalAlpha = 0.4;
  for (const run of state.runs) {
    line(
      box.pen, frame,
      run.points.map((p) => p.voltage),
      run.points.map((p) => p.value),
      BLUE
    );
  }
  box.pen.globalAlpha = 1;
  line(
    box.pen, frame,
    state.points.map((p) => p.voltage),
    state.points.map((p) => p.value),
    GREEN,
    { dots: true }
  );
  showRuns();
}

function showRuns() {
  const note = el("runs-note");
  note.textContent = "";
  for (const run of state.runs) {
    const entry = document.createElement("span");
    entry.innerHTML = '<i class="swatch" style="background:#1f6feb;opacity:0.4"></i>';
    entry.appendChild(document.createTextNode(run.label));
    note.appendChild(entry);
  }
}

// What one request asks for, flattened to names and values. Used only to say
// how two runs differ, which is a comparison of the requests and not of any
// physics: the page is reading back what it sent.
function settings(body) {
  const flat = {
    device: body.device.kind,
    sweep: body.sweep.kind,
    contact: body.sweep.contact,
    voltages: body.sweep.voltages.join(" "),
  };
  const sources = [body.device.parameters, body.sweep.settings, body.sweep.models];
  for (const source of sources) {
    for (const [name, value] of Object.entries(source || {})) flat[name] = value;
  }
  return flat;
}

function show(value) {
  const number = Number(value);
  return typeof value === "number" || (value !== "" && isFinite(number))
    ? format(number)
    : String(value);
}

// This run described by what it does not share with another one. An overlay
// is labelled against the run that came after it, so the label says what made
// this curve the one it is.
function differences(mine, other) {
  if (!other) return "the run on screen";
  const a = settings(mine), b = settings(other);
  const changed = Object.keys(a).filter((k) => String(a[k]) !== String(b[k]));
  if (!changed.length) return "the same settings";
  return changed.map((k) => k + " " + show(a[k])).join(", ");
}

// The run on screen becomes an overlay, and every overlay is relabelled
// against the run that follows it, the newest against `next`.
function keep(next) {
  if (state.curve && state.points.length && state.request) {
    state.runs.push({
      points: state.points.slice(),
      request: state.request,
      label: "",
    });
  }
  for (let i = 0; i < state.runs.length; i++) {
    const after = i + 1 < state.runs.length ? state.runs[i + 1].request : next;
    state.runs[i].label = differences(state.runs[i].request, after);
  }
}

function clearRuns() {
  state.runs = [];
  drawCurve();
  showRuns();
}

function drawProfile() {
  const box = fit(el("profile"));
  const fields = state.fields;
  if (!fields) return;

  if (fields.shape.length === 2) {
    el("cutline-panel").hidden = false;
    drawImage(box, fields);
    if (el("streamlines").checked && fields.arrays.Jx) {
      drawStreamlines(box, fields, traceStreamlines(fields, 12, 6));
    }
    // A cutline already drawn follows the point slider to the new state.
    if (state.cutline) drawCutline(fields, state.cutline.from, state.cutline.to);
    return;
  }
  const banded = el("bands").checked;
  el("legend-bands").hidden = !banded;
  if (banded) {
    drawBands(box, fields.arrays.x, (name) => fields.arrays[name]);
    return;
  }
  const xs = fields.arrays.x;
  const potential = axes(box.pen, box, xs, fields.arrays.psi, {
    logY: false, twin: true, ink: BLUE,
  });
  line(box.pen, potential, xs, fields.arrays.psi, BLUE);

  // The densities span decades, so they share a log frame over the same x.
  const both = Array.from(fields.arrays.n).concat(Array.from(fields.arrays.p));
  const carriers = axes(box.pen, box, xs, both, {
    logY: true, twin: true, right: true, ink: GREEN,
  });
  line(box.pen, carriers, xs, fields.arrays.n, GREEN);
  line(box.pen, carriers, xs, fields.arrays.p, RED);
}

function drawImage(box, fields) {
  const ny = fields.shape[0], nx = fields.shape[1];
  const values = fields.arrays.psi;
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!isFinite(lo)) return;
  if (lo === hi) hi = lo + 1;

  const clamp = (t) => Math.min(1, Math.max(0, t));
  const image = box.pen.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const t = (values[j * nx + i] - lo) / (hi - lo);
      // y[0] is the bottom of the substrate and a canvas counts rows from
      // the top, so row j is painted at ny - 1 - j and the gate is up.
      const at = 4 * ((ny - 1 - j) * nx + i);
      image.data[at] = Math.round(255 * clamp(1.5 * t));
      image.data[at + 1] = Math.round(255 * clamp(1.5 * t - 0.25));
      image.data[at + 2] = Math.round(255 * clamp(1.6 - 1.5 * t));
      image.data[at + 3] = 255;
    }
  }
  const sheet = document.createElement("canvas");
  sheet.width = nx;
  sheet.height = ny;
  sheet.getContext("2d").putImageData(image, 0, 0);
  box.pen.imageSmoothingEnabled = true;
  box.pen.drawImage(sheet, 0, 0, box.width, box.height);
}

// ----------------------------------------------------------------- the form

function knob(parameter) {
  const label = document.createElement("label");
  const name = document.createElement("span");
  name.textContent = parameter.name;
  label.appendChild(name);
  name.appendChild(explainButton(() => explain(parameter.topic, parameter)));

  let input;
  if (parameter.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(parameter.default);
  } else if (parameter.choices && parameter.choices.length) {
    input = document.createElement("select");
    for (const choice of parameter.choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      input.appendChild(option);
    }
    input.value = String(parameter.default);
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = String(parameter.default);
  }
  input.dataset.name = parameter.name;
  input.dataset.kind = parameter.type;

  const control = document.createElement("span");
  control.className = "control";
  control.appendChild(input);
  const drag = slider(parameter, input);
  if (drag) control.appendChild(drag);
  label.appendChild(control);
  return label;
}

// A knob with a range declared in its own docstring gets a slider over that
// range. A knob without one gets the box alone: the page has no business
// inventing ends for a span it was told nothing about.
function slider(parameter, box) {
  if (parameter.low === null || parameter.high === null) return null;
  const log = parameter.axis === "log";
  const at = (value) => (log ? decades(value) : value);

  const drag = document.createElement("input");
  drag.type = "range";
  drag.dataset.slider = parameter.name;
  drag.min = String(at(parameter.low));
  drag.max = String(at(parameter.high));
  // A whole number knob steps by one, everything else by a two hundredth of
  // its travel, which is finer than the slider has pixels.
  drag.step = String(
    !log && parameter.type === "int"
      ? 1
      : (at(parameter.high) - at(parameter.low)) / 200
  );
  drag.value = String(at(parameter.default));

  drag.addEventListener("input", () => {
    const raw = Number(drag.value);
    const value = log ? undecades(raw) : raw;
    box.value =
      parameter.type === "int" ? String(Math.round(value)) : format(value);
    nudge();
  });
  return drag;
}

function fill(container, parameters) {
  container.textContent = "";
  for (const parameter of parameters) container.appendChild(knob(parameter));
}

function collect(container) {
  const sent = {};
  for (const input of container.querySelectorAll("[data-name]")) {
    const name = input.dataset.name;
    if (input.dataset.kind === "bool") {
      sent[name] = input.checked;
    } else if (input.dataset.kind === "str") {
      sent[name] = input.value;
    } else if (input.value.trim() !== "") {
      const value = Number(input.value);
      if (!isFinite(value)) {
        throw new Error(name + " is not a number: " + input.value);
      }
      sent[name] = input.dataset.kind === "int" ? Math.round(value) : value;
    }
  }
  return sent;
}

function defaultContact() {
  return el("device-kind").value === "pn_diode" ? "anode" : "gate";
}

function onDeviceKind() {
  const kind = el("device-kind").value;
  fill(el("device-knobs"), state.schema.devices[kind]);
  el("contact").value = defaultContact();

  // A 1D device solves while you drag it. A 2D one is seconds to minutes, so
  // it keeps the solve button and is offered a coarse mesh instead, and the
  // page says which it is rather than leaving a student to find out.
  el("mesh-choice").hidden = !state.schema.presets[kind];
  el("mesh-note").textContent = "";
  el("live-note").textContent = live()
    ? "moving a slider re-solves this device."
    : "press solve: this device has two axes and takes seconds to minutes.";
}

function live() {
  return state.schema.dimensions[el("device-kind").value] === 1;
}

// The coarse mesh, or back to the one the constructor declares. Only the
// knobs the preset names are touched, so a doping a student set stays set.
function useMesh(coarse) {
  const kind = el("device-kind").value;
  const preset = state.schema.presets[kind];
  if (!preset) return;
  const defaults = {};
  for (const parameter of state.schema.devices[kind]) {
    defaults[parameter.name] = parameter.default;
  }
  for (const name of Object.keys(preset.parameters)) {
    const input = el("device-knobs").querySelector('[data-name="' + name + '"]');
    if (input) {
      input.value = String(coarse ? preset.parameters[name] : defaults[name]);
    }
  }
  el("mesh-note").textContent = coarse
    ? preset.note
    : "The mesh this device is validated on.";
}

function onSweepKind() {
  const kind = el("sweep-kind").value;
  fill(el("sweep-knobs"), state.schema.sweeps[kind]);
  el("measure-at").parentElement.style.display = kind === "transfer" ? "" : "none";
  el("model-knobs").parentElement.style.display = kind === "cv" ? "none" : "";
  el("contact").value = defaultContact();
  state.valueName = kind === "cv" ? "capacitance" : "current";
}

// -------------------------------------------------------------- the solving

async function schema() {
  state.schema = await (await fetch("/api/schema")).json();

  for (const kind of Object.keys(state.schema.devices)) {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = kind;
    el("device-kind").appendChild(option);
  }
  for (const kind of Object.keys(state.schema.sweeps)) {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = kind;
    el("sweep-kind").appendChild(option);
  }
  fill(el("model-knobs"), state.schema.models);
  onDeviceKind();
  onSweepKind();
  el("state").textContent = "ready";
  el("drawer-close").addEventListener("click", () => el("drawer").classList.remove("open"));
  markExplainable(document, state.schema.plots);
  wireCutline();
}

function voltages() {
  const asked = el("voltages").value.split(/[,\s]+/).filter((s) => s.length);
  const values = asked.map(Number);
  if (!values.length) throw new Error("no voltages asked for");
  if (values.some((v) => !isFinite(v))) {
    throw new Error("the voltage list has something in it that is not a number");
  }
  return values;
}

function request() {
  const kind = el("sweep-kind").value;
  const sweep = {
    kind: kind,
    contact: el("contact").value.trim(),
    voltages: voltages(),
    settings: collect(el("sweep-knobs")),
  };
  if (kind !== "cv") sweep.models = collect(el("model-knobs"));
  if (kind === "transfer" && el("measure-at").value.trim()) {
    sweep.measure_at = el("measure-at").value.trim();
  }
  return {
    device: {
      kind: el("device-kind").value,
      parameters: collect(el("device-knobs")),
    },
    sweep: sweep,
  };
}

function clear() {
  state.residual = [];
  state.newton = null;
  state.stalled = null;
  state.rejected = [];
  state.points = [];
  state.curve = null;
  state.fields = null;
  state.cutline = null;
  el("cutline-panel").hidden = true;
  el("message").textContent = "";
  el("curve-note").textContent = "no points yet";
  el("residual-note").textContent = "";
  el("profile-note").textContent = "";
  el("point").disabled = true;
  el("point").max = "0";
  drawResidual();
  drawCurve();
  drawProfile();
}

async function solve() {
  let body;
  try {
    body = request();
  } catch (problem) {
    el("message").textContent = String(problem.message || problem);
    return;
  }
  // Before anything is cleared: whatever finished is now an overlay.
  keep(body);
  state.request = body;
  clear();

  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const failure = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    el("message").textContent = failure.detail || JSON.stringify(failure);
    el("state").textContent = "refused";
    return;
  }

  state.job = (await response.json()).id;
  el("solve").disabled = true;
  el("cancel").disabled = false;
  el("state").textContent = "solving";
  listen();
}

function listen() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(
    scheme + "://" + location.host + "/api/jobs/" + state.job + "/stream"
  );
  state.socket = socket;
  socket.onmessage = (event) => frame(JSON.parse(event.data));
  socket.onclose = () => {
    el("solve").disabled = false;
    el("cancel").disabled = true;
  };
}

// The equation family carrying the largest entry of a split. A null is a
// number that was not finite, which is the largest there is.
function largest(families) {
  let worst = "", size = -Infinity;
  for (const [family, value] of Object.entries(families)) {
    const measured = value === null ? Infinity : value;
    if (measured > size) { size = measured; worst = family; }
  }
  return worst;
}

// Which equation a failed Newton attempt was stuck in. Both the residual and
// the update are named, because an attempt fails when either misses its
// tolerance and the page does not know the tolerances. Measured on a stalled
// MOSFET: every residual sat at 1e-14 and the n update at 1.8e-10, so naming
// the residual alone would have pointed at the wrong test.
function stalled(split) {
  if (!split || !split.residual) return "";
  let named = "largest residual in " + largest(split.residual);
  if (split.update) named += ", largest update in " + largest(split.update);
  return " (" + named + ")";
}

function frame(body) {
  if (body.type === "newton") {
    state.residual.push({
      value: body.residual,
      colour: BLUE,
      families: body.residual_by_family,
    });
    if (body.residual_by_family) {
      state.newton = {
        residual: body.residual_by_family,
        update: body.update_by_family,
      };
    }
    el("residual-note").textContent =
      "iteration " + body.iteration + (body.limited ? ", step limited" : "");
    drawResidual();
  } else if (body.type === "gummel") {
    state.residual.push({ value: body.update, colour: PURPLE });
    el("residual-note").textContent = "cycle " + body.iteration;
    drawResidual();
  } else if (body.type === "continuation") {
    if (!body.converged) {
      state.rejected.push(state.residual.length - 1);
      el("residual-note").textContent =
        "rejected at " + format(body.parameter) + " V, " + body.message +
        stalled(state.newton);
      state.stalled = state.newton;
      drawResidual();
    }
  } else if (body.type === "point" || body.type === "cv_point") {
    const voltage = body.type === "point" ? body.voltage : body.gate_voltage;
    const value = body.type === "point" ? body.current : body.capacitance;
    state.points.push({ voltage: voltage, value: value });
    el("curve-note").textContent =
      state.points.length + " points, last at " + format(voltage) + " V";
    drawCurve();
  } else if (body.type === "status") {
    finish(body);
  }
}

async function finish(body) {
  el("state").textContent = body.status;
  const statusTopic = state.schema.statuses[body.status];
  if (statusTopic) {
    el("state").appendChild(explainButton(() => explain(statusTopic)));
  }
  if (body.message) el("message").textContent = body.message;
  if (body.status === "failed") {
    el("message").textContent += stalled(state.stalled || state.newton);
  }
  if (body.dropped) {
    el("curve-note").textContent += ", " + body.dropped + " frames dropped";
  }
  if (body.status !== "done") return;

  const response = await fetch("/api/jobs/" + state.job + "/result");
  if (!response.ok) return;
  state.curve = await response.json();
  state.points = state.curve.points.map((point) => ({
    voltage: point.voltage,
    value: point[state.valueName],
  }));
  drawCurve();
  if (!state.curve.complete && state.curve.message) {
    el("message").textContent = state.curve.message + stalled(state.stalled);
  }
  if (state.points.length) {
    el("point").disabled = false;
    el("point").max = String(state.points.length - 1);
    el("point").value = String(state.points.length - 1);
    await profile(state.points.length - 1);
  }
}

function readFields(buffer) {
  const view = new DataView(buffer);
  const length = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, length))
  );
  const payload = new Float32Array(buffer, 4 + length);
  const arrays = {};
  let at = 0;
  for (const array of header.arrays) {
    arrays[array.name] = payload.subarray(at, at + array.length);
    at += array.length;
  }
  return {
    shape: header.shape,
    index: header.index,
    voltage: header.voltage,
    arrays: arrays,
  };
}

async function profile(index) {
  const response = await fetch("/api/jobs/" + state.job + "/fields/" + index);
  if (!response.ok) return;
  state.fields = readFields(await response.arrayBuffer());
  el("profile-note").textContent = "at " + format(state.fields.voltage) + " V";
  drawProfile();
}

async function cancel() {
  if (!state.job) return;
  const response = await fetch("/api/jobs/" + state.job + "/cancel", {
    method: "POST",
  });
  const body = await response.json();
  el("state").textContent = body.cancelled ? "cancelling" : "already finished";
}

// A slider moved. The page waits for the hand to stop, then replaces the
// solve in flight rather than adding to it, so a drag across a knob leaves
// one job running however many positions it passed through.
function nudge() {
  if (!live()) return;
  clearTimeout(state.pending);
  state.pending = setTimeout(() => {
    state.pending = null;
    stop().then(solve);
  }, LIVE_DELAY);
}

// Put down the job on screen without saying anything about it. The socket
// goes first, so the frames of a solve nobody is waiting for any more stop
// reaching the plots, and the cancel lands at that solve's next iteration.
async function stop() {
  const socket = state.socket;
  if (socket) {
    state.socket = null;
    socket.onclose = null;
    socket.onmessage = null;
    socket.close();
  }
  if (state.job) {
    await fetch("/api/jobs/" + state.job + "/cancel", { method: "POST" });
  }
}

el("device-kind").addEventListener("change", onDeviceKind);
el("sweep-kind").addEventListener("change", onSweepKind);
el("solve").addEventListener("click", solve);
el("cancel").addEventListener("click", cancel);
el("clear-runs").addEventListener("click", clearRuns);
el("mesh-coarse").addEventListener("click", () => useMesh(true));
el("mesh-converged").addEventListener("click", () => useMesh(false));
el("residual-log").addEventListener("change", drawResidual);
el("curve-log").addEventListener("change", drawCurve);
el("bands").addEventListener("change", drawProfile);
el("streamlines").addEventListener("change", drawProfile);
el("point").addEventListener("change", (event) =>
  profile(Number(event.target.value))
);
window.addEventListener("resize", () => {
  drawResidual();
  drawCurve();
  drawProfile();
});

schema().catch((problem) => {
  el("state").textContent = "the schema could not be loaded";
  el("message").textContent = String(problem.message || problem);
});
