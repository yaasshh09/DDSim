"use strict";

const PAD = { left: 70, right: 18, twin: 70, top: 14, bottom: 42 };
const PLOTS = ["residual", "curve", "profile", "cutline"];

const plotFont = (size, weight) => (weight || 400) + " " + size + "px " + token("sans");
const crisp = (v) => Math.round(v) + 0.5;

function niceStep(span, count) {
  const raw = span / count;
  const size = undecades(Math.floor(decades(raw)));
  const ratio = raw / size;
  return (ratio < 1.5 ? 1 : ratio < 3 ? 2 : ratio < 7 ? 5 : 10) * size;
}

function linearTicks(lo, hi, step) {
  const mantissa = Math.round(step / undecades(Math.floor(decades(step) + 1e-9)));
  const minor = step / (mantissa === 2 ? 4 : 5);
  const major = [], minors = [];
  for (let k = Math.ceil(lo / minor - 1e-6); k * minor <= hi + minor * 1e-6; k++) {
    const v = k * minor;
    const whole = Math.abs(v / step - Math.round(v / step)) < 1e-6;
    if (whole) major.push(Math.round(v / step) * step); else minors.push(v);
  }
  return { major: major, minor: minors, step: step };
}

function logTicks(lo, hi, room) {
  const a = Math.round(decades(lo)), b = Math.round(decades(hi));
  const every = Math.max(1, Math.ceil((b - a) / Math.max(2, Math.floor(room / 26))));
  const major = [], minor = [];
  for (let d = a; d <= b; d++) {
    if ((d - a) % every === 0) major.push(d); else minor.push(undecades(d));
    if (b - a <= 6 && d < b) {
      for (let k = 2; k <= 9; k++) minor.push(k * undecades(d));
    }
  }
  return { major: major, minor: minor };
}

function tickLabel(v, step) {
  if (Math.abs(v) < step * 1e-6) return "0";
  const size = Math.abs(v);
  if (size >= 1e4 || step < 1e-3) {
    const digits = Math.floor(decades(size) + 1e-9) - Math.floor(decades(step) + 1e-9);
    return v.toExponential(Math.max(0, Math.min(4, digits)))
      .replace(/(\.\d*?)0+e/, "$1e").replace(".e", "e");
  }
  return v.toFixed(Math.max(0, -Math.floor(decades(step) + 1e-9)));
}

function precise(v) {
  if (v === null || v === undefined || !isFinite(v)) return "none";
  if (v === 0) return "0";
  const size = Math.abs(v);
  return size >= 1e4 || size < 1e-3 ? v.toExponential(3) : String(Number(v.toPrecision(4)));
}

function power(pen, exponent, x, y, align) {
  pen.font = plotFont(11);
  const base = pen.measureText("10").width;
  pen.font = plotFont(8.5);
  const up = pen.measureText(String(exponent)).width;
  const total = base + up + 1;
  const left = align === "right" ? x - total : align === "center" ? x - total / 2 : x;
  pen.textAlign = "left";
  pen.font = plotFont(11);
  pen.fillText("10", left, y);
  pen.font = plotFont(8.5);
  pen.fillText(String(exponent), left + base + 1, y - 5);
}

function bounds(xs, ys, logY, view) {
  let xlo = Infinity, xhi = -Infinity;
  for (const x of xs) {
    if (!isFinite(x)) continue;
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
  }
  if (view) { xlo = view.xlo; xhi = view.xhi; }
  if (!isFinite(xlo)) { xlo = 0; xhi = 1; }
  if (xlo === xhi) xhi = xlo + 1;

  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < ys.length; k++) {
    const y = ys[k];
    if (y === null || !isFinite(y) || (logY && !(y > 0))) continue;
    if (view && xs.length) {
      const x = xs[k % xs.length];
      if (x < xlo || x > xhi) continue;
    }
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  if (!isFinite(lo)) { lo = logY ? 0.1 : 0; hi = logY ? 10 : 1; }
  if (lo === hi) { lo = logY ? lo / 10 : lo - 1; hi = logY ? hi * 10 : hi + 1; }
  return { xlo: xlo, xhi: xhi, lo: lo, hi: hi };
}

function axes(pen, box, xs, ys, options) {
  options = options || {};
  const logY = Boolean(options.logY);
  const right = Boolean(options.right);
  const canvas = pen.canvas;
  const inner = {
    x: PAD.left,
    y: PAD.top,
    width: Math.max(10, box.width - PAD.left - (options.twin ? PAD.twin : PAD.right)),
    height: Math.max(10, box.height - PAD.top - PAD.bottom),
  };
  const range = bounds(xs, ys, logY, canvas._view);
  const xlo = range.xlo, xhi = range.xhi;
  let lo = range.lo, hi = range.hi, yTicks;
  if (logY) {
    const a = Math.floor(decades(lo) + 1e-9);
    const b = Math.max(a + 1, Math.ceil(decades(hi) - 1e-9));
    lo = undecades(a);
    hi = undecades(b);
    yTicks = logTicks(lo, hi, inner.height);
  } else {
    const step = niceStep(hi - lo, Math.max(3, Math.round(inner.height / 42)));
    lo = Math.floor(lo / step + 1e-9) * step;
    hi = Math.ceil(hi / step - 1e-9) * step;
    yTicks = linearTicks(lo, hi, step);
  }
  const xStep = niceStep(xhi - xlo, Math.max(3, Math.round(inner.width / 90)));
  const xTicks = linearTicks(xlo, xhi, xStep);

  const yOf = (y) => {
    const t = logY
      ? (decades(y) - decades(lo)) / (decades(hi) - decades(lo))
      : (y - lo) / (hi - lo);
    return inner.y + inner.height - t * inner.height;
  };
  const xOf = (x) => inner.x + ((x - xlo) / (xhi - xlo)) * inner.width;
  const xFrom = (px) => xlo + ((px - inner.x) / inner.width) * (xhi - xlo);
  const bottom = inner.y + inner.height;
  const edge = right ? inner.x + inner.width : inner.x;
  const out = right ? 1 : -1;

  const strokes = (colour, width, draw) => {
    pen.strokeStyle = colour;
    pen.lineWidth = width;
    pen.beginPath();
    draw();
    pen.stroke();
  };
  const ys2 = yTicks.minor.map(yOf);
  const ysMajor = logY ? yTicks.major.map((d) => yOf(undecades(d))) : yTicks.major.map(yOf);

  if (!right) {
    strokes(ink("--grid-minor"), 1, () => {
      for (const v of xTicks.minor) { pen.moveTo(crisp(xOf(v)), inner.y); pen.lineTo(crisp(xOf(v)), bottom); }
      for (const y of ys2) { pen.moveTo(inner.x, crisp(y)); pen.lineTo(inner.x + inner.width, crisp(y)); }
    });
    strokes(ink("--grid"), 1, () => {
      for (const v of xTicks.major) { pen.moveTo(crisp(xOf(v)), inner.y); pen.lineTo(crisp(xOf(v)), bottom); }
      for (const y of ysMajor) { pen.moveTo(inner.x, crisp(y)); pen.lineTo(inner.x + inner.width, crisp(y)); }
    });
    if (xlo < 0 && xhi > 0) {
      strokes(ink("--line-bright"), 1, () => { pen.moveTo(crisp(xOf(0)), inner.y); pen.lineTo(crisp(xOf(0)), bottom); });
    }
    if (!logY && lo < 0 && hi > 0) {
      strokes(ink("--line-bright"), 1, () => { pen.moveTo(inner.x, crisp(yOf(0))); pen.lineTo(inner.x + inner.width, crisp(yOf(0))); });
    }
    strokes(ink("--line-bright"), 1, () =>
      pen.rect(crisp(inner.x), crisp(inner.y), Math.round(inner.width), Math.round(inner.height)));
    strokes(ink("--ink-3"), 1, () => {
      for (const v of xTicks.major) { pen.moveTo(crisp(xOf(v)), bottom); pen.lineTo(crisp(xOf(v)), bottom + 5); }
      for (const v of xTicks.minor) { pen.moveTo(crisp(xOf(v)), bottom); pen.lineTo(crisp(xOf(v)), bottom + 3); }
    });
    pen.fillStyle = ink("--ink-3");
    pen.font = plotFont(11);
    pen.textAlign = "center";
    for (const v of xTicks.major) pen.fillText(tickLabel(v, xStep), xOf(v), bottom + 17);
    if (options.xTitle) {
      pen.fillStyle = ink("--ink");
      pen.font = plotFont(11.5, 600);
      pen.fillText(options.xTitle, inner.x + inner.width / 2, box.height - 6);
    }
  }

  const colour = options.ink || ink("--ink-3");
  strokes(colour, 1, () => {
    for (const y of ysMajor) { pen.moveTo(edge, crisp(y)); pen.lineTo(edge + out * 5, crisp(y)); }
    for (const y of ys2) { pen.moveTo(edge, crisp(y)); pen.lineTo(edge + out * 3, crisp(y)); }
  });
  pen.fillStyle = colour;
  pen.font = plotFont(11);
  pen.textAlign = right ? "left" : "right";
  const labelX = edge + out * 8;
  if (logY) {
    yTicks.major.forEach((d, k) => power(pen, d, labelX, ysMajor[k] + 4, right ? "left" : "right"));
  } else {
    yTicks.major.forEach((v, k) => pen.fillText(tickLabel(v, yTicks.step), labelX, ysMajor[k] + 4));
  }
  if (options.yTitle) {
    pen.save();
    pen.translate(right ? box.width - 10 : 13, inner.y + inner.height / 2);
    pen.rotate(right ? Math.PI / 2 : -Math.PI / 2);
    pen.textAlign = "center";
    pen.fillStyle = options.ink || ink("--ink");
    pen.font = plotFont(11.5, 600);
    pen.fillText(options.yTitle, 0, 0);
    pen.restore();
  }

  const frame = {
    xOf: xOf, yOf: yOf, xFrom: xFrom, inner: inner, logY: logY,
    xTitle: options.xTitle || "x", yTitle: options.yTitle || "y",
  };
  if (!right) canvas._frame = frame;
  return frame;
}

function line(pen, frame, xs, ys, colour, options) {
  options = options || {};
  const usable = (y) => y !== null && isFinite(y) && !(frame.logY && !(y > 0));
  const inner = frame.inner;
  pen.save();
  pen.beginPath();
  pen.rect(inner.x, inner.y, inner.width, inner.height);
  pen.clip();
  pen.strokeStyle = colour;
  pen.lineWidth = options.width || 1.6;
  pen.lineJoin = "round";
  pen.setLineDash(options.dash || []);
  pen.beginPath();
  let down = false;
  for (let i = 0; i < xs.length; i++) {
    if (!usable(ys[i])) { down = false; continue; }
    const px = frame.xOf(xs[i]), py = frame.yOf(ys[i]);
    if (down) pen.lineTo(px, py); else pen.moveTo(px, py);
    down = true;
  }
  pen.stroke();
  pen.setLineDash([]);

  if (options.dots) {
    pen.fillStyle = colour;
    for (let i = 0; i < xs.length; i++) {
      if (!usable(ys[i])) continue;
      pen.beginPath();
      pen.arc(frame.xOf(xs[i]), frame.yOf(ys[i]), options.dots === true ? 2.5 : options.dots, 0, 2 * Math.PI);
      pen.fill();
    }
  }
  pen.restore();
  pen.canvas._series.push({
    frame: frame, xs: xs, ys: ys, raw: options.raw || ys,
    colour: colour, name: options.name || "", errors: options.errors || null,
  });
}

function whiskers(pen, frame, xs, ys, errors, colour) {
  const inner = frame.inner;
  pen.save();
  pen.beginPath();
  pen.rect(inner.x, inner.y, inner.width, inner.height);
  pen.clip();
  pen.strokeStyle = colour;
  pen.lineWidth = 1.2;
  for (let i = 0; i < xs.length; i++) {
    const e = errors[i], y = ys[i];
    if (e === null || e === undefined || !isFinite(e) || y === null || !isFinite(y)) continue;
    const top = y + Math.abs(e), low = y - Math.abs(e);
    if (frame.logY && !(top > 0)) continue;
    const px = frame.xOf(xs[i]), a = frame.yOf(top);
    const b = frame.yOf(frame.logY && !(low > 0) ? y : low);
    pen.beginPath();
    pen.moveTo(px, a);
    pen.lineTo(px, b);
    pen.moveTo(px - 3, a);
    pen.lineTo(px + 3, a);
    pen.moveTo(px - 3, b);
    pen.lineTo(px + 3, b);
    pen.stroke();
  }
  pen.restore();
}

function ring(pen, frame, x, y, colour) {
  pen.strokeStyle = colour;
  pen.lineWidth = 1.5;
  pen.beginPath();
  pen.arc(frame.xOf(x), frame.yOf(y), 4.5, 0, 2 * Math.PI);
  pen.stroke();
}

function colourBar(box, lo, hi, title) {
  const pen = box.pen;
  const width = Math.min(170, box.width * 0.35);
  const x = box.width - width - 14, y = 10;
  pen.fillStyle = token("ground-veil");
  pen.fillRect(x - 10, y - 5, width + 20, 46);
  for (let k = 0; k < width; k++) {
    const c = ramp(k / (width - 1));
    pen.fillStyle = "rgb(" + c.join(",") + ")";
    pen.fillRect(x + k, y, 1.5, 10);
  }
  pen.strokeStyle = token("line-bright");
  pen.lineWidth = 1;
  pen.strokeRect(crisp(x), crisp(y), Math.round(width), 10);
  const step = niceStep(hi - lo, 4);
  const ticks = linearTicks(lo, hi, step).major;
  pen.fillStyle = token("ink");
  pen.font = plotFont(10);
  pen.textAlign = "center";
  for (const v of ticks) {
    const px = x + ((v - lo) / (hi - lo)) * width;
    pen.fillRect(Math.round(px), y + 10, 1, 3);
    pen.fillText(tickLabel(v, step), px, y + 23);
  }
  pen.font = plotFont(10, 600);
  pen.fillText(title, x + width / 2, y + 36);
}

const tip = document.createElement("div");
tip.id = "plot-tip";
tip.hidden = true;
document.body.appendChild(tip);

function nearest(series, x) {
  let best = -1, gap = Infinity;
  for (let k = 0; k < series.xs.length; k++) {
    const y = series.ys[k];
    if (y === null || !isFinite(y) || (series.frame.logY && !(y > 0))) continue;
    const d = Math.abs(series.xs[k] - x);
    if (d < gap) { gap = d; best = k; }
  }
  return best;
}

function overlay(canvas) {
  const pen = canvas.getContext("2d");
  if (!canvas._base) canvas._base = pen.getImageData(0, 0, canvas.width, canvas.height);
  else pen.putImageData(canvas._base, 0, 0);
  return pen;
}

function hideTip(canvas) {
  tip.hidden = true;
  if (canvas._base) canvas.getContext("2d").putImageData(canvas._base, 0, 0);
}

function showTip(event, title, rows) {
  tip.textContent = "";
  const head = document.createElement("b");
  head.textContent = title;
  tip.appendChild(head);
  for (const row of rows) {
    const item = document.createElement("span");
    if (row.colour) {
      const swatch = document.createElement("i");
      swatch.style.background = row.colour;
      item.appendChild(swatch);
    }
    item.appendChild(document.createTextNode(row.text));
    tip.appendChild(item);
  }
  tip.hidden = false;
  const left = event.clientX + 16 + tip.offsetWidth > window.innerWidth
    ? event.clientX - 16 - tip.offsetWidth : event.clientX + 16;
  tip.style.left = Math.max(4, left) + "px";
  tip.style.top = Math.max(4, event.clientY + 14) + "px";
}

function shorten(text) {
  return text.length > 48 ? text.slice(0, 47) + "…" : text;
}

function hoverImage(canvas, event) {
  const fields = canvas._image;
  const box = { width: canvas.clientWidth, height: Number(canvas.dataset.height) };
  const at = indexAt(box, fields, event.offsetX, event.offsetY);
  const ny = fields.shape[0], nx = fields.shape[1];
  const along = (axis, index, size) => {
    const k = Math.max(0, Math.min(size - 2, Math.floor(index)));
    return axis[k] + (index - k) * (axis[k + 1] - axis[k]);
  };
  const x = along(fields.arrays.x, at.i, nx) * 1e4;
  const y = along(fields.arrays.y, at.j, ny) * 1e4;
  const rows = [];
  for (const name of ["psi", "n", "p", "Ec", "Efn", "Efp"]) {
    if (!fields.arrays[name]) continue;
    const unit = fields.units && fields.units[name]
      ? " " + fields.units[name].replace("^-3", "⁻³").replace("^2", "²") : "";
    rows.push({ text: name + " = " + precise(interpolate(fields, name, at.i, at.j)) + unit });
  }
  if (fields.arrays.Jx) {
    const size = Math.hypot(interpolate(fields, "Jx", at.i, at.j), interpolate(fields, "Jy", at.i, at.j));
    rows.push({ text: "|J| = " + precise(size) + " A/cm²" });
  }
  const pen = overlay(canvas);
  pen.strokeStyle = token("streamline-halo");
  pen.lineWidth = 1;
  pen.beginPath();
  pen.moveTo(crisp(event.offsetX), 0);
  pen.lineTo(crisp(event.offsetX), box.height);
  pen.moveTo(0, crisp(event.offsetY));
  pen.lineTo(box.width, crisp(event.offsetY));
  pen.stroke();
  showTip(event, "x = " + precise(x) + " µm, y = " + precise(y) + " µm", rows);
}

function hover(canvas, event) {
  if (canvas._image) return hoverImage(canvas, event);
  const frame = canvas._frame;
  if (!frame || !canvas._series.length) return hideTip(canvas);
  const inner = frame.inner;
  const px = event.offsetX;
  if (px < inner.x || px > inner.x + inner.width) return hideTip(canvas);
  const x = frame.xFrom(px);
  const pen = overlay(canvas);

  if (canvas._drag !== null && canvas._drag !== undefined) {
    pen.fillStyle = token("signal-wash");
    pen.fillRect(Math.min(canvas._drag, px), inner.y, Math.abs(px - canvas._drag), inner.height);
  }

  const rows = [];
  let snap = null;
  for (const series of canvas._series) {
    const k = nearest(series, x);
    if (k < 0) continue;
    const sx = series.frame.xOf(series.xs[k]);
    if (Math.abs(sx - px) > 40) continue;
    if (snap === null) snap = { px: sx, x: series.xs[k] };
    const sy = series.frame.yOf(series.ys[k]);
    pen.fillStyle = series.colour;
    pen.beginPath();
    pen.arc(sx, sy, 4, 0, 2 * Math.PI);
    pen.fill();
    pen.strokeStyle = token("panel");
    pen.lineWidth = 1.5;
    pen.stroke();
    rows.push({
      colour: series.colour,
      text: shorten(series.name) + ": " + precise(series.raw[k]) +
        (series.errors && series.errors[k] !== null && series.errors[k] !== undefined
          ? " ± " + precise(Math.abs(series.errors[k])) : "") +
        (series.xs[k] !== snap.x ? " at " + precise(series.xs[k]) : ""),
    });
  }
  if (!snap) return hideTip(canvas);
  pen.strokeStyle = token("line-bright");
  pen.lineWidth = 1;
  pen.setLineDash([3, 3]);
  pen.beginPath();
  pen.moveTo(crisp(snap.px), inner.y);
  pen.lineTo(crisp(snap.px), inner.y + inner.height);
  pen.stroke();
  pen.setLineDash([]);
  showTip(event, frame.xTitle + " = " + precise(snap.x), rows);
}

function redraw(id) {
  if (id === "residual") drawResidual();
  else if (id === "curve") drawCurve();
  else if (id === "profile") drawProfile();
  else if (state.cutline && state.fields) {
    drawCutline(state.fields, state.cutline.from, state.cutline.to);
  }
}

function save(name, href) {
  const link = document.createElement("a");
  link.href = href;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function seriesTable(canvas) {
  const frame = canvas._frame;
  if (!frame || !canvas._series.length) return "";
  const quote = (text) => '"' + String(text).replace(/"/g, '""') + '"';
  const rows = ["series," + quote(frame.xTitle) + ",value"];
  for (const series of canvas._series) {
    for (let k = 0; k < series.xs.length; k++) {
      const y = series.raw[k];
      if (y === null || !isFinite(y)) continue;
      rows.push(quote(series.name) + "," + series.xs[k] + "," + y);
    }
  }
  return rows.join("\n") + "\n";
}

function fieldsTable(fields) {
  if (!fields) return "";
  const flat = fields.shape.length === 1;
  const names = Object.keys(fields.arrays).filter((name) => name !== "x" && name !== "y");
  const head = (name) =>
    name + (fields.units && fields.units[name] ? " [" + fields.units[name] + "]" : "");
  const columns = (flat ? ["x"] : ["x", "y"]).concat(names);
  const rows = [columns.map(head).join(",")];
  const nx = flat ? fields.shape[0] : fields.shape[1], ny = flat ? 1 : fields.shape[0];
  const cell = (v) => (isFinite(v) ? Number(v).toPrecision(7) : "");
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const at = j * nx + i;
      const cells = flat ? [fields.arrays.x[i]] : [fields.arrays.x[i], fields.arrays.y[j]];
      for (const name of names) cells.push(fields.arrays[name][at]);
      rows.push(cells.map(cell).join(","));
    }
  }
  return rows.join("\n") + "\n";
}

function saveCsv(id) {
  const text = id === "profile" ? fieldsTable(state.fields) : seriesTable(el(id));
  if (!text) return;
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  save("ddsim-" + id + ".csv", url);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function savePng(id) {
  const canvas = el(id);
  hideTip(canvas);
  save("ddsim-" + id + ".png", canvas.toDataURL("image/png"));
}

for (const id of PLOTS) {
  const canvas = el(id);
  canvas._drag = null;
  canvas.addEventListener("mousemove", (event) => hover(canvas, event));
  canvas.addEventListener("mouseleave", () => {
    canvas._drag = null;
    hideTip(canvas);
  });
  canvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || canvas._image || !canvas._frame) return;
    canvas._drag = event.offsetX;
  });
  canvas.addEventListener("mouseup", (event) => {
    const from = canvas._drag;
    canvas._drag = null;
    if (from === null || !canvas._frame || Math.abs(event.offsetX - from) < 6) return;
    const a = canvas._frame.xFrom(from), b = canvas._frame.xFrom(event.offsetX);
    canvas._view = { xlo: Math.min(a, b), xhi: Math.max(a, b) };
    tip.hidden = true;
    redraw(id);
  });
  canvas.addEventListener("dblclick", () => {
    if (!canvas._view) return;
    canvas._view = null;
    tip.hidden = true;
    redraw(id);
  });
  const png = el(id + "-png"), csv = el(id + "-csv");
  if (png) png.addEventListener("click", () => savePng(id));
  if (csv) csv.addEventListener("click", () => saveCsv(id));
}
