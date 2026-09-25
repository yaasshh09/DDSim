"use strict";

const BLUE = "#5fd4d6", PURPLE = "#a992ef", GREEN = "#e3a74f", RED = "#ea7a68";

const el = (id) => document.getElementById(id);

const LIVE_DELAY = 200;

const state = {
  schema: null,
  job: null,
  socket: null,
  residual: [],
  newton: null,
  stalled: null,
  rejected: [],
  points: [],
  curve: null,
  fields: null,
  cutline: null,
  valueName: "current",
  runs: [],
  request: null,
  pending: null,
  turn: 0,
};

function fit(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  if (!canvas.dataset.height) canvas.dataset.height = canvas.getAttribute("height");
  const height = Number(canvas.dataset.height);
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
  pen.font = "500 10px Poppins, system-ui, sans-serif";
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

function drawResidual() {
  const box = fit(el("residual"));
  if (!state.residual.length) return;
  const logY = el("residual-log").checked;
  const xs = state.residual.map((_, i) => i);
  const ys = state.residual.flatMap((r) =>
    r.families ? Object.values(r.families).concat([r.value]) : [r.value]
  );
  const frame = axes(box.pen, box, xs, ys, { logY: logY });

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

function drawCurve() {
  const box = fit(el("curve"));
  if (!state.points.length && !state.runs.length) return;

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

function settings(body) {
  const flat = {
    device: body.device.kind,
    sweep: body.sweep.kind,
    contact: body.sweep.contact,
    voltages: body.sweep.voltages.join(" "),
  };
  const sources = [body.device.parameters, body.sweep.settings, body.sweep.models];
  for (const source of sources) {
    for (const [name, value] of Object.entries(source || {})) {
      if (name === "regions") {
        flat[name] = value
          .map((r) => r.dopant + " " + show(r.length) + " cm " + show(r.concentration))
          .join(", ");
      } else if (Array.isArray(value)) {
        flat[name] = value.map((r) => Object.values(r).map(show).join(" ")).join(", ");
      } else {
        flat[name] = value;
      }
    }
  }
  return flat;
}

function show(value) {
  const number = Number(value);
  return typeof value === "number" || (value !== "" && isFinite(number))
    ? format(number)
    : String(value);
}

function differences(mine, other) {
  if (!other) return "the run on screen";
  const a = settings(mine), b = settings(other);
  const changed = Object.keys(a).filter((k) => String(a[k]) !== String(b[k]));
  if (!changed.length) return "the same settings";
  return changed.map((k) => k + " " + show(a[k])).join(", ");
}

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
    el("legend-bands").hidden = true;
    drawImage(box, fields);
    let note = "at " + format(fields.voltage) + " V";
    if (el("streamlines").checked && fields.arrays.Jx) {
      const lines = traceStreamlines(fields, 12, 6);
      drawStreamlines(box, fields, lines);
      if (!lines.length) note += ", no current flows: every contact is at the same voltage";
    }
    el("profile-note").textContent = note;
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

  const both = Array.from(fields.arrays.n).concat(Array.from(fields.arrays.p));
  const carriers = axes(box.pen, box, xs, both, {
    logY: true, twin: true, right: true, ink: GREEN,
  });
  line(box.pen, carriers, xs, fields.arrays.n, GREEN);
  line(box.pen, carriers, xs, fields.arrays.p, RED);
}

const RAMP = [
  [19, 34, 39],
  [43, 111, 116],
  [95, 212, 214],
];

function ramp(t) {
  const span = t < 0.5 ? 0 : 1;
  const within = t * 2 - span;
  const low = RAMP[span], high = RAMP[span + 1];
  return [
    Math.round(low[0] + (high[0] - low[0]) * within),
    Math.round(low[1] + (high[1] - low[1]) * within),
    Math.round(low[2] + (high[2] - low[2]) * within),
  ];
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
      const t = clamp((values[j * nx + i] - lo) / (hi - lo));
      const colour = ramp(t);
      const at = 4 * ((ny - 1 - j) * nx + i);
      image.data[at] = colour[0];
      image.data[at + 1] = colour[1];
      image.data[at + 2] = colour[2];
      image.data[at + 3] = 255;
    }
  }
  const sheet = document.createElement("canvas");
  sheet.width = nx;
  sheet.height = ny;
  sheet.getContext("2d").putImageData(image, 0, 0);
  box.pen.imageSmoothingEnabled = true;
  box.pen.drawImage(sheet, 0.5, 0.5, nx - 1, ny - 1, 0, 0, box.width, box.height);
}

function start(parameter) {
  return "value" in parameter ? parameter.value : parameter.default;
}

function withValues(parameters, values) {
  return parameters.map((p) =>
    p.name in values ? Object.assign({}, p, { value: values[p.name] }) : p
  );
}

function setUp(body, meshNote) {
  const sweep = body.sweep;
  putDevice(body.device);
  el("sweep-kind").value = sweep.kind;
  onSweepKind();
  fill(el("sweep-knobs"),
    withValues(state.schema.sweeps[sweep.kind], sweep.settings || {}));
  fill(el("model-knobs"), withValues(state.schema.models, sweep.models || {}));
  el("contact").value = sweep.contact;
  el("measure-at").value = sweep.measure_at || "";
  el("voltages").value = sweep.voltages.join(", ");
  if (meshNote) el("mesh-note").textContent = meshNote;
}

function putDevice(device) {
  const parameters = device.parameters || {};
  el("device-kind").value = device.kind;
  onDeviceKind();
  fill(el("device-knobs"), withValues(state.schema.devices[device.kind], parameters));
  if (parameters.regions) showRegions(parameters.regions);
  if (parameters.blocks) showDrawing(parameters);
}

function knob(parameter) {
  const label = document.createElement("label");
  const name = document.createElement("span");
  name.textContent = parameter.label;
  name.appendChild(explainButton(() => explain(parameter.topic, parameter)));
  const symbol = document.createElement("i");
  symbol.className = "symbol";
  symbol.textContent = parameter.name + (parameter.unit ? " " + parameter.unit : "");
  name.appendChild(symbol);
  label.appendChild(name);

  let input;
  if (parameter.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(start(parameter));
  } else if (parameter.choices && parameter.choices.length) {
    input = document.createElement("select");
    for (const choice of parameter.choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      input.appendChild(option);
    }
    input.value = String(start(parameter));
  } else {
    input = document.createElement("input");
    input.type = "text";
    const first = start(parameter);
    input.value =
      parameter.type === "float" && typeof first === "number"
        ? format(first)
        : String(first);
  }
  input.dataset.name = parameter.name;
  input.dataset.kind = parameter.type;

  const control = document.createElement("span");
  control.className = "control";
  control.appendChild(input);
  const drag = slider(parameter, input);
  if (drag) control.appendChild(drag);
  label.appendChild(control);

  if (parameter.type === "int" || parameter.type === "float") {
    input.dataset.good = input.value;
    input.addEventListener("change", () => {
      const value = Number(input.value);
      if (input.value.trim() === "" || !isFinite(value)) {
        input.value = input.dataset.good;
        return;
      }
      input.value = written(parameter, within(parameter, value));
      if (drag) drag.value = String(position(parameter, Number(input.value)));
      settle(parameter, input, drag);
    });
  }
  return label;
}

const position = (parameter, value) =>
  parameter.axis !== "log" ? value
    : parameter.low < 0 ? -decades(-value) : decades(value);
const unposition = (parameter, at) =>
  parameter.axis !== "log" ? at
    : parameter.low < 0 ? -undecades(-at) : undecades(at);
const within = (parameter, value) =>
  parameter.low === null || parameter.high === null
    ? value
    : Math.min(parameter.high, Math.max(parameter.low, value));
const written = (parameter, value) =>
  parameter.type === "int" ? String(Math.round(value)) : format(value);

function slider(parameter, box) {
  if (parameter.low === null || parameter.high === null) return null;
  const log = parameter.axis === "log";

  const drag = document.createElement("input");
  drag.type = "range";
  drag.dataset.slider = parameter.name;
  drag.min = String(position(parameter, parameter.low));
  drag.max = String(position(parameter, parameter.high));
  drag.step = String(
    !log && parameter.type === "int"
      ? 1
      : (Number(drag.max) - Number(drag.min)) / 200
  );
  drag.value = String(position(parameter, start(parameter)));

  drag.addEventListener("input", () => {
    box.value = written(parameter, unposition(parameter, Number(drag.value)));
    settle(parameter, box, drag);
  });
  return drag;
}

async function refusal() {
  let body;
  try {
    body = JSON.stringify(deviceRequest());
  } catch (problem) {
    return "";
  }
  const response = await fetch("/api/devices/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body,
  });
  if (response.ok) return "";
  const failure = await response.json().catch(() => ({ detail: response.statusText }));
  return String(failure.detail || "this device can't be built");
}

async function holdBuildable(parameter, box, drag) {
  const why = await refusal();
  if (!why) return "";
  const asked = box.value;
  let inside = box.dataset.good;
  let near = position(parameter, Number(inside));
  let far = position(parameter, Number(asked));
  for (let step = 0; step < 12; step++) {
    const middle = (near + far) / 2;
    const tried = written(parameter, unposition(parameter, middle));
    if (tried === inside || tried === box.value) break;
    box.value = tried;
    if (await refusal()) {
      far = middle;
    } else {
      near = middle;
      inside = tried;
    }
  }
  box.value = inside;
  if (drag) drag.value = String(position(parameter, Number(inside)));
  return (
    parameter.label + " stops at " + inside +
    (parameter.unit && parameter.unit !== "1" ? " " + parameter.unit : "") +
    ", the furthest it can go with the other settings as they are. " +
    "Why: " + why
  );
}

function settle(parameter, box, drag) {
  clearTimeout(state.pending);
  state.pending = setTimeout(async () => {
    state.pending = null;
    const turn = ++state.turn;
    if (box.closest("#device-knobs")) {
      const note = await holdBuildable(parameter, box, drag);
      if (turn !== state.turn) return;
      el("knob-note").textContent = note;
    }
    box.dataset.good = box.value;
    if (live()) stop().then(solve);
  }, LIVE_DELAY);
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
        throw new Error(name + " isn't a number: " + input.value);
      }
      sent[name] = input.dataset.kind === "int" ? Math.round(value) : value;
    }
  }
  return sent;
}

function defaultContact() {
  const contacts = { pn_diode: "anode", stack: "left" };
  return contacts[el("device-kind").value] || "gate";
}

function offerContacts() {
  let names = state.schema.contacts[el("device-kind").value] || [];
  try {
    const drawn = collectDrawing();
    if (drawn) names = drawn.electrodes.map((electrode) => electrode.name);
  } catch (problem) {
    return;
  }
  for (const [id, blank] of [["contact", null], ["measure-at", "default"]]) {
    const select = el(id);
    const kept = select.value;
    select.textContent = "";
    if (blank !== null) select.appendChild(new Option(blank, ""));
    for (const name of names) select.appendChild(new Option(name, name));
    if (Array.from(select.options).some((option) => option.value === kept)) {
      select.value = kept;
    }
  }
}

function onDeviceKind() {
  const kind = el("device-kind").value;
  fill(el("device-knobs"), state.schema.devices[kind]);
  showRegions(state.schema.regions[kind] || null);
  showDrawing(state.schema.drawings[kind] || null);
  offerContacts();
  el("contact").value = defaultContact();

  el("mesh-choice").hidden = !state.schema.presets[kind];
  el("mesh-note").textContent = "";
  el("knob-note").textContent = "";
  el("voltage-note").textContent = "";
  el("bands").parentElement.style.display = live() ? "inline-flex" : "none";
  el("live-note").textContent = live()
    ? "Move a slider and this device solves again straight away."
    : "Press solve when you're ready. This is a 2D device, so it takes seconds to minutes.";
}

function live() {
  return state.schema.dimensions[el("device-kind").value] === 1;
}

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
    : "The mesh this device was validated on.";
}

function onSweepKind() {
  const kind = el("sweep-kind").value;
  fill(el("sweep-knobs"), state.schema.sweeps[kind]);
  el("measure-at").parentElement.style.display = kind === "transfer" ? "" : "none";
  el("model-knobs").parentElement.style.display = kind === "cv" ? "none" : "";
  el("contact").value = defaultContact();
  state.valueName = kind === "cv" ? "capacitance" : "current";
}

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
  el("node-budget").textContent = String(state.schema.node_budget);
  onDeviceKind();
  onSweepKind();
  welcome();
  el("state").textContent = "ready";
  el("drawer-close").addEventListener("click", () => el("drawer").classList.remove("open"));
  markExplainable(document, state.schema.plots);
  wireCutline();
}

function voltages() {
  const asked = el("voltages").value.split(/[,\s]+/).filter((s) => s.length);
  const values = asked.map(Number);
  if (!values.length) throw new Error("the voltage list is empty");
  if (values.some((v) => !isFinite(v))) {
    throw new Error("something in the voltage list isn't a number");
  }
  const contact = el("contact").value.trim();
  const bias = state.schema.devices[el("device-kind").value]
    .find((p) => p.name === contact + "_voltage" && p.low !== null);
  if (!bias) return values;
  const held = values.map((v) => within(bias, v));
  if (held.some((v, i) => v !== values[i])) {
    el("voltages").value = held.join(", ");
    el("voltage-note").textContent =
      "held to " + format(bias.low) + " V to " + format(bias.high) +
      " V, the range allowed for the " + contact + " bias";
  }
  return held;
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
  return { device: deviceRequest(), sweep: sweep };
}

function deviceRequest() {
  const parameters = collect(el("device-knobs"));
  const regions = collectRegions();
  if (regions) parameters.regions = regions;
  Object.assign(parameters, collectDrawing() || {});
  return { kind: el("device-kind").value, parameters: parameters };
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
  el("curve-note").textContent = "a point lands here as each voltage is solved";
  el("residual-note").textContent = "fills in while a solve is running";
  el("profile-note").textContent = "the inside of the device, once it's solved";
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
  keep(body);
  state.request = body;
  clear();
  el("built-note").hidden =
    !state.schema.regions[body.device.kind] && !state.schema.drawings[body.device.kind];

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

function largest(families) {
  let worst = "", size = -Infinity;
  for (const [family, value] of Object.entries(families)) {
    const measured = value === null ? Infinity : value;
    if (measured > size) { size = measured; worst = family; }
  }
  return worst;
}

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

  setMode("results");

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

function chooseDevice() {
  onDeviceKind();
  const sweep = el("sweep-kind").value;
  const wanted = live() ? (sweep === "transfer" ? "iv" : sweep)
    : (sweep === "iv" ? "transfer" : sweep);
  if (wanted !== sweep) {
    el("sweep-kind").value = wanted;
    onSweepKind();
  }
}

el("device-kind").addEventListener("change", chooseDevice);
el("sweep-kind").addEventListener("change", onSweepKind);
el("contact").addEventListener("focus", offerContacts);
el("measure-at").addEventListener("focus", offerContacts);
el("voltages").addEventListener("input", () => { el("voltage-note").textContent = ""; });
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
  el("state").textContent = "couldn't load the device list";
  el("message").textContent = String(problem.message || problem);
});
