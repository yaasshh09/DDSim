const PARTS = {
  blocks: [
    { field: "material", choices: ["silicon", "oxide"] },
    { field: "x0", title: "left [cm]" },
    { field: "x1", title: "right [cm]" },
    { field: "y0", title: "bottom [cm]" },
    { field: "y1", title: "top [cm]" },
  ],
  implants: [
    { field: "dopant", choices: ["p", "n"] },
    { field: "concentration", title: "peak doping [cm^-3]" },
    { field: "x0", title: "left [cm]" },
    { field: "x1", title: "right [cm]" },
    { field: "y0", title: "bottom [cm]" },
    { field: "y1", title: "top [cm]" },
    { field: "profile", choices: ["uniform", "gaussian"] },
    { field: "straggle", title: "gaussian: depth sigma below and above [cm]" },
    { field: "lateral", title: "gaussian: erfc length past the sides [cm]" },
  ],
  electrodes: [
    { field: "name", text: true, title: "terminal name, which a sweep uses" },
    { field: "kind", choices: ["ohmic", "gate"] },
    { field: "x0", title: "start across [cm]" },
    { field: "x1", title: "end across [cm], x0 for a vertical one" },
    { field: "y0", title: "start up [cm]" },
    { field: "y1", title: "end up [cm], y0 for a horizontal one" },
    { field: "voltage", title: "bias [V]" },
    { field: "work_function", title: "gate work function [eV]" },
  ],
};

function partRow(part, record) {
  const row = document.createElement("div");
  row.className = "region part";
  for (const spec of PARTS[part]) {
    let input;
    if (spec.choices) {
      input = document.createElement("select");
      for (const choice of spec.choices) {
        const option = document.createElement("option");
        option.value = choice;
        option.textContent = choice;
        input.appendChild(option);
      }
      input.value = record[spec.field];
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.value = spec.text ? record[spec.field] : exact(record[spec.field]);
    }
    input.title = spec.title || spec.field;
    input.placeholder = spec.field;
    input.dataset.part = spec.field;
    input.addEventListener("change", drawPreview);
    row.appendChild(input);
  }
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "flat";
  remove.textContent = "x";
  remove.title = "remove this one";
  remove.addEventListener("click", () => { row.remove(); drawPreview(); });
  row.appendChild(remove);
  return row;
}

function showDrawing(parts) {
  el("drawing").hidden = !parts;
  setMode(parts ? "build" : "results");
  for (const part of Object.keys(PARTS)) {
    const list = el("drawing-" + part);
    list.textContent = "";
    for (const record of (parts && parts[part]) || []) {
      list.appendChild(partRow(part, record));
    }
  }
  if (parts) drawPreview();
}

function collectDrawing() {
  if (el("drawing").hidden) return null;
  const parts = {};
  for (const [part, specs] of Object.entries(PARTS)) {
    parts[part] = Array.from(el("drawing-" + part).children, (row, index) => {
      const record = {};
      for (const spec of specs) {
        const text = row.querySelector("[data-part=" + spec.field + "]").value;
        if (spec.choices || spec.text) {
          record[spec.field] = text;
          continue;
        }
        const value = Number(text);
        if (text.trim() === "" || !isFinite(value)) {
          throw new Error(part.slice(0, -1) + " " + (index + 1) + ": " +
            spec.field + " is not a number: " + text);
        }
        record[spec.field] = value;
      }
      return record;
    });
  }
  return parts;
}

function drawingSoFar() {
  try {
    return collectDrawing();
  } catch (problem) {
    return null;
  }
}

const SNAP = 6;

function snapped(value, edges, scale) {
  let best = null, gap = SNAP;
  for (const edge of edges) {
    const distance = Math.abs(scale(edge) - scale(value));
    if (distance <= gap) { best = edge; gap = distance; }
  }
  return best === null ? Number(value.toPrecision(4)) : best;
}

function defaultWorkFunction() {
  const parts = state.schema.drawings[el("device-kind").value];
  return parts.electrodes.length ? parts.electrodes[0].work_function : null;
}

function addFromDrag(from, to) {
  const parts = drawingSoFar();
  if (!parts) return;
  const box = fit(el("drawing-view"));
  const at = previewFrame(box, parts);
  const xs = [0, at.size.width], ys = [0, at.size.height];
  for (const list of Object.values(parts)) {
    for (const r of list) { xs.push(r.x0, r.x1); ys.push(r.y0, r.y1); }
  }
  const a = at.back(from.x, from.y), b = at.back(to.x, to.y);
  const x = [a.x, b.x].map((v) => snapped(v, xs, at.x)).sort((first, second) => first - second);
  const y = [a.y, b.y].map((v) => snapped(v, ys, at.y)).sort((first, second) => first - second);
  const what = el("drawing-tool").value;
  let part, record;
  if (what === "silicon" || what === "oxide") {
    part = "blocks";
    record = { material: what, x0: x[0], x1: x[1], y0: y[0], y1: y[1] };
  } else if (what === "n" || what === "p") {
    part = "implants";
    record = { dopant: what, concentration: 1e18, x0: x[0], x1: x[1],
      y0: y[0], y1: y[1], profile: "uniform", straggle: 0, lateral: 0 };
  } else {
    part = "electrodes";
    const across = Math.abs(to.x - from.x) >= Math.abs(to.y - from.y);
    const start = { x: snapped(a.x, xs, at.x), y: snapped(a.y, ys, at.y) };
    record = {
      name: what + (el("drawing-electrodes").children.length + 1),
      kind: what,
      x0: across ? x[0] : start.x, x1: across ? x[1] : start.x,
      y0: across ? start.y : y[0], y1: across ? start.y : y[1],
      voltage: 0, work_function: defaultWorkFunction(),
    };
  }
  el("drawing-" + part).appendChild(partRow(part, record));
  drawPreview();
}

function blank(part) {
  const size = extent(drawingSoFar() || { blocks: [] });
  const whole = { x0: 0, x1: size.width, y0: 0, y1: size.height };
  if (part === "blocks") return Object.assign({ material: "silicon" }, whole);
  if (part === "implants") {
    return Object.assign({ dopant: "p", concentration: 1e16,
      profile: "uniform", straggle: 0, lateral: 0 }, whole);
  }
  return { name: "contact", kind: "ohmic", x0: 0, x1: size.width, y0: 0, y1: 0,
    voltage: 0, work_function: defaultWorkFunction() };
}

function wireDrawing() {
  buildEditor();
  const view = el("drawing-view");
  let from = null;
  const point = (event) => {
    const rect = view.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };
  view.addEventListener("mousedown", (event) => { from = point(event); });
  view.addEventListener("mouseup", (event) => {
    if (!from) return;
    const to = point(event);
    if (Math.hypot(to.x - from.x, to.y - from.y) > SNAP) addFromDrag(from, to);
    from = null;
  });
  for (const part of Object.keys(PARTS)) {
    el("drawing-add-" + part).addEventListener("click", () => {
      const rows = drawingSoFar();
      const list = rows ? rows[part] : [];
      const last = list.length ? list[list.length - 1] : null;
      el("drawing-" + part).appendChild(partRow(part, last || blank(part)));
      drawPreview();
    });
  }
}

wireDrawing();
