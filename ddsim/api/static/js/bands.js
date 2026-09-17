"use strict";

// The band diagram view and the 2D cutline. Every energy was computed on the
// server. What happens here is geometry: choosing arrays to draw and, for a
// cutline, bilinear interpolation of a node array along a line in the image.

function drawBands(box, xs, pick) {
  const Ec = pick("Ec"), Ev = pick("Ev"), Efn = pick("Efn"), Efp = pick("Efp");
  const all = Array.from(Ec).concat(Array.from(Ev), Array.from(Efn), Array.from(Efp));
  const frame = axes(box.pen, box, xs, all, { logY: false });
  line(box.pen, frame, xs, Ec, BLUE);
  line(box.pen, frame, xs, Ev, BLUE);
  line(box.pen, frame, xs, Efn, GREEN);
  line(box.pen, frame, xs, Efp, RED);
}

// Bilinear interpolation in index space. (i, j) are fractional column and row
// indices, the space the image and the cutline share. An oxide node carries
// NaN, so a sample that touches one is NaN too and the line breaks there.
function interpolate(fields, name, i, j) {
  const ny = fields.shape[0], nx = fields.shape[1];
  const values = fields.arrays[name];
  const i0 = Math.max(0, Math.min(nx - 2, Math.floor(i)));
  const j0 = Math.max(0, Math.min(ny - 2, Math.floor(j)));
  const u = i - i0, v = j - j0;
  const at = (a, b) => values[b * nx + a];
  return (1 - u) * (1 - v) * at(i0, j0) + u * (1 - v) * at(i0 + 1, j0) +
    (1 - u) * v * at(i0, j0 + 1) + u * v * at(i0 + 1, j0 + 1);
}

// Distance along the cutline in cm, from the physical coordinates of each
// sample, so the band plot's x axis is a real length.
function sampleAlong(fields, name, from, to, count) {
  const xAxis = fields.arrays.x, yAxis = fields.arrays.y;
  const ny = fields.shape[0], nx = fields.shape[1];
  const position = (axis, index, size) => {
    const k = Math.max(0, Math.min(size - 2, Math.floor(index)));
    return axis[k] + (index - k) * (axis[k + 1] - axis[k]);
  };
  const x0 = position(xAxis, from.i, nx), y0 = position(yAxis, from.j, ny);
  const s = new Float64Array(count), values = new Float64Array(count);
  for (let k = 0; k < count; k++) {
    const t = k / (count - 1);
    const i = from.i + t * (to.i - from.i), j = from.j + t * (to.j - from.j);
    s[k] = Math.hypot(position(xAxis, i, nx) - x0, position(yAxis, j, ny) - y0);
    values[k] = interpolate(fields, name, i, j);
  }
  return { s: s, values: values };
}

// Canvas pixel to fractional (i, j). drawImage paints row 0 at the bottom, so
// j is flipped the same way here.
function indexAt(box, fields, px, py) {
  const ny = fields.shape[0], nx = fields.shape[1];
  return { i: (px / box.width) * (nx - 1), j: (1 - py / box.height) * (ny - 1) };
}

function drawCutline(fields, from, to) {
  const box = fit(el("cutline"));
  const count = 200;
  const s = sampleAlong(fields, "Ec", from, to, count).s;
  drawBands(box, s, (name) => sampleAlong(fields, name, from, to, count).values);
}

function wireCutline() {
  const canvas = el("profile");
  const sized = () => ({
    width: canvas.clientWidth,
    height: Number(canvas.getAttribute("height")),
  });
  let start = null;
  canvas.addEventListener("mousedown", (event) => {
    if (!state.fields || state.fields.shape.length !== 2) return;
    start = indexAt(sized(), state.fields, event.offsetX, event.offsetY);
  });
  canvas.addEventListener("mouseup", (event) => {
    if (!start || !state.fields) return;
    state.cutline = {
      from: start,
      to: indexAt(sized(), state.fields, event.offsetX, event.offsetY),
    };
    start = null;
    drawCutline(state.fields, state.cutline.from, state.cutline.to);
    el("cutline-note").textContent = "bands along the line you drew";
  });
}
