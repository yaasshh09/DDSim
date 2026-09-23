"use strict";

const MATERIAL_FILL = { silicon: "#2d4a52", oxide: "#16252a" };
const DOPANT_INK = { n: "#e3a74f", p: "#5fd4d6" };

const DOPANT_WASH = { n: "rgba(227, 167, 79, 0.26)", p: "rgba(95, 212, 214, 0.16)" };

function extent(parts) {
  let width = 0, height = 0;
  for (const block of parts.blocks) {
    width = Math.max(width, block.x1);
    height = Math.max(height, block.y1);
  }
  return { width: width || 1, height: height || 1 };
}

const MIN_BAND = 30;

function bands(parts, height) {
  const edges = new Set([0, height]);
  for (const list of Object.values(parts)) {
    for (const r of list) {
      if (r.y0 >= 0 && r.y0 <= height) edges.add(r.y0);
      if (r.y1 >= 0 && r.y1 <= height) edges.add(r.y1);
    }
  }
  return [...edges].sort((first, second) => first - second);
}

function stretch(edges, height, pixels) {
  const spans = edges.length - 1;
  if (spans < 1) return { at: () => 0, back: () => 0 };
  const spare = Math.max(0, pixels - spans * MIN_BAND);
  const tops = [0];
  for (let i = 0; i < spans; i++) {
    const share = height > 0 ? (edges[i + 1] - edges[i]) / height : 0;
    tops.push(tops[i] + MIN_BAND + spare * share);
  }
  return {
    at: (v) => {
      if (v <= edges[0]) return tops[0];
      for (let i = 0; i < spans; i++) {
        if (v <= edges[i + 1]) {
          const width = edges[i + 1] - edges[i];
          const into = width > 0 ? (v - edges[i]) / width : 0;
          return tops[i] + into * (tops[i + 1] - tops[i]);
        }
      }
      return tops[spans];
    },
    back: (py) => {
      if (py <= tops[0]) return edges[0];
      for (let i = 0; i < spans; i++) {
        if (py <= tops[i + 1]) {
          const run = tops[i + 1] - tops[i];
          const into = run > 0 ? (py - tops[i]) / run : 0;
          return edges[i] + into * (edges[i + 1] - edges[i]);
        }
      }
      return edges[spans];
    },
  };
}

function previewFrame(box, parts) {
  const size = extent(parts);
  const up = stretch(bands(parts, size.height), size.height, box.height);
  return {
    x: (v) => (v / size.width) * box.width,
    y: (v) => box.height - up.at(v),
    back: (px, py) => ({
      x: (px / box.width) * size.width,
      y: up.back(box.height - py),
    }),
    size: size,
    stretched: bands(parts, size.height).length > 2,
  };
}

function drawPreview() {
  const parts = drawingSoFar();
  const box = fit(el("drawing-view"));
  if (!parts) return;
  const at = previewFrame(box, parts);
  const pen = box.pen;
  for (const block of parts.blocks) {
    pen.fillStyle = MATERIAL_FILL[block.material] || "#0b1417";
    pen.fillRect(at.x(block.x0), at.y(block.y1),
      at.x(block.x1) - at.x(block.x0), at.y(block.y0) - at.y(block.y1));
  }
  pen.setLineDash([4, 3]);
  for (const implant of parts.implants) {
    const left = at.x(implant.x0), top = at.y(implant.y1);
    const wide = at.x(implant.x1) - left, tall = at.y(implant.y0) - top;
    pen.fillStyle = DOPANT_WASH[implant.dopant] || "rgba(255, 255, 255, 0.1)";
    pen.fillRect(left, top, wide, tall);
    pen.strokeStyle = DOPANT_INK[implant.dopant] || "#0b1417";
    pen.strokeRect(left, top, wide, tall);
  }
  pen.setLineDash([]);
  pen.lineWidth = 4;
  pen.font = "500 11px Poppins, system-ui, sans-serif";
  for (const electrode of parts.electrodes) {
    pen.strokeStyle = electrode.kind === "gate" ? "#a992ef" : "#e8f1f2";
    pen.beginPath();
    pen.moveTo(at.x(electrode.x0), at.y(electrode.y0));
    pen.lineTo(at.x(electrode.x1), at.y(electrode.y1));
    pen.stroke();
    label(pen, box, electrode.name, pen.strokeStyle,
      Math.min(at.x(electrode.x0) + 3, box.width - 46),
      Math.max(at.y(electrode.y1) - 5, 11));
  }
  pen.lineWidth = 1;
  showScale(at.size, at.stretched);
}

function label(pen, box, text, colour, x, y) {
  const width = pen.measureText(text).width;
  pen.fillStyle = "rgba(11, 20, 23, 0.72)";
  pen.fillRect(x - 3, y - 10, width + 6, 13);
  pen.fillStyle = colour;
  pen.fillText(text, x, y);
}

function showScale(size, stretched) {
  const bar = el("build-scale");
  if (!bar) return;
  bar.innerHTML =
    '<span><i class="swatch block-si"></i>silicon</span>' +
    '<span><i class="swatch block-ox"></i>oxide</span>' +
    '<span><i class="swatch dope-n"></i>n doping</span>' +
    '<span><i class="swatch dope-p"></i>p doping</span>' +
    '<span><i class="swatch gate"></i>gate</span>' +
    '<span><i class="swatch ohmic"></i>ohmic</span>';
  const extent = document.createElement("span");
  extent.className = "extent";
  extent.textContent =
    "x 0 to " + format(size.width) + " cm  ·  y 0 to " +
    format(size.height) + " cm" + (stretched ? "  ·  y stretched" : "");
  bar.appendChild(extent);
}
