"use strict";

const PART_ABOUT = {
  blocks: {
    title: "Shapes",
    lede:
      "Rectangles of silicon or oxide. Each new one paints over the ones " +
      "before it, so oxide drawn on top of silicon cuts a trench. Together " +
      "they have to fill a rectangle with no gaps.",
    add: "add a shape",
  },
  implants: {
    title: "Doping",
    lede:
      "Rectangles of added impurity, p or n. Where two overlap they add up. " +
      "Uniform stays at its peak inside the box and stops dead at the edge. " +
      "Gaussian fades out past the edge, the way a real implant does.",
    add: "add doping",
  },
  electrodes: {
    title: "Contacts",
    lede:
      "Straight lines where a wire meets the device. An ohmic contact sits " +
      "on silicon. A gate sits on oxide and has a work function. Whatever " +
      "you name it shows up in the sweep's contact box.",
    add: "add a contact",
  },
};

const PART_LABELS = {
  material: "material",
  dopant: "type",
  concentration: "peak doping",
  name: "name",
  kind: "kind",
  x0: "left",
  x1: "right",
  y0: "bottom",
  y1: "top",
  profile: "profile",
  straggle: "depth spread",
  lateral: "side spread",
  voltage: "volts",
  work_function: "work fn",
};

const BUILD_HINT =
  "Pick a tool, then drag a rectangle on the picture. If you let go near an " +
  "edge you've already drawn, it snaps onto it. A contact runs along " +
  "whichever way you dragged further. Everything you draw shows up as a row " +
  "below, and those numbers are the real device. The picture stretches its " +
  "two axes differently so a thin oxide stays visible.";

const BUILD_EMPTY =
  "This is where you draw your own device. Choose \"drawing\" as the device " +
  "kind on the left, or pick \"Draw your own\" on the start screen, and the " +
  "editor opens here.";

const DRAWING_NOTE =
  "Rectangles only. The mesh is a grid of lines that each run the whole " +
  "width or height, and a cell that's half oxide and half silicon wouldn't " +
  "have one permittivity. Meshes over <b id=\"node-budget\"></b> nodes get " +
  "turned down. Results on a drawn device are the validated solver's answer " +
  "on a structure nothing has checked, and the page says so under the curve.";

function buildEditor() {
  const into = el("build-parts");
  if (!into || into.children.length) return;
  for (const [part, about] of Object.entries(PART_ABOUT)) {
    const group = document.createElement("div");
    group.className = "partGroup";

    const heading = document.createElement("div");
    heading.className = "section";
    heading.textContent = about.title;
    group.appendChild(heading);

    const lede = document.createElement("p");
    lede.className = "aside lede";
    lede.textContent = about.lede;
    group.appendChild(lede);

    group.appendChild(partHeadings(part));

    const rows = document.createElement("div");
    rows.id = "drawing-" + part;
    group.appendChild(rows);

    const add = document.createElement("button");
    add.type = "button";
    add.className = "flat";
    add.id = "drawing-add-" + part;
    add.textContent = about.add;
    const row = document.createElement("div");
    row.className = "row";
    row.appendChild(add);
    group.appendChild(row);

    into.appendChild(group);
  }
  el("build-hint").textContent = BUILD_HINT;
  el("build-empty").textContent = BUILD_EMPTY;
  el("drawing-note").innerHTML = DRAWING_NOTE;
}

function partHeadings(part) {
  const head = document.createElement("div");
  head.className = "region part headings";
  for (const spec of PARTS[part]) {
    const cell = document.createElement("span");
    cell.textContent = PART_LABELS[spec.field] || spec.field;
    cell.title = spec.title || spec.field;
    cell.dataset.heading = spec.field;
    head.appendChild(cell);
  }
  return head;
}

function setMode(mode) {
  const building = mode === "build";
  el("build").hidden = !building;
  el("plots").hidden = building;
  el("mode-build").setAttribute("aria-selected", String(building));
  el("mode-results").setAttribute("aria-selected", String(!building));
}

el("mode-build").addEventListener("click", () => setMode("build"));
el("mode-results").addEventListener("click", () => setMode("results"));
