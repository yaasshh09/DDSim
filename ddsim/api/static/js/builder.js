"use strict";

// The device editor's chrome and the Build / Results switch.
//
// drawing.js owns the rows and the drag. This owns the frame around them: the
// three groups, the words above each one, the column headings that turn a
// line of unlabelled boxes into a table, and which half of the stage is on
// screen. Kept apart from drawing.js because that file is already long, and
// because none of this knows anything about geometry.

// What each part is, in words. The three lists are the whole vocabulary of a
// drawing, so this is where a newcomer finds out what one is.
const PART_ABOUT = {
  blocks: {
    title: "Shapes",
    lede:
      "Rectangles of silicon or oxide. A later one is painted over an " +
      "earlier one, so an oxide drawn on top of silicon cuts a trench in it. " +
      "Together they have to cover a rectangle with no gaps.",
    add: "add a shape",
  },
  implants: {
    title: "Doping",
    lede:
      "Rectangles of added impurity, p or n. Where two overlap they add up. " +
      "Uniform holds its peak inside the box and stops at the edge; gaussian " +
      "fades out past it, the way a real implant does.",
    add: "add doping",
  },
  electrodes: {
    title: "Contacts",
    lede:
      "Straight lines where a wire meets the device. An ohmic contact sits " +
      "on silicon. A gate sits on oxide and has a work function. The name is " +
      "what the sweep's contact box takes.",
    add: "add a contact",
  },
};

// One heading per field, in the order drawing.js lays the row out.
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
  "Pick a tool, then drag a rectangle on the picture. A drag that ends near " +
  "an edge already drawn snaps onto it. A contact is the longer way you " +
  "dragged. Everything you draw shows up as a row below, and the numbers in " +
  "those rows are the device: the picture stretches its two axes apart so a " +
  "thin oxide stays visible.";

const BUILD_EMPTY =
  "This is where a device you draw is put together. Choose \"drawing\" as " +
  "the device kind on the left, or pick \"Draw your own\" from the start " +
  "screen, and the editor opens here.";

const DRAWING_NOTE =
  "Rectangles only, because the mesh is a grid of lines that each run the " +
  "whole width or height, and a cell that is part oxide and part silicon " +
  "has no single permittivity. A mesh over <b id=\"node-budget\"></b> nodes " +
  "is refused. A device you drew is the validated solver's answer on a " +
  "structure nothing has checked, and the page says so under the curve.";

// The three groups, each a heading, a line of explanation, a row of column
// names, the rows themselves and a button that adds one.
function buildEditor() {
  const into = el("build-parts");
  if (!into || into.children.length) return;
  for (const [part, about] of Object.entries(PART_ABOUT)) {
    const group = document.createElement("div");
    group.className = "part-group";

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

// A row of column names above the boxes, sharing .part so the widths line up
// with the rows underneath it.
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

// Which half of the stage is showing. The rails do not move: the knobs and
// the solve button stay reachable while a device is being drawn, because
// drawing one and setting its mesh are the same job.
function setMode(mode) {
  const building = mode === "build";
  el("build").hidden = !building;
  el("plots").hidden = building;
  el("mode-build").setAttribute("aria-selected", String(building));
  el("mode-results").setAttribute("aria-selected", String(!building));
}

el("mode-build").addEventListener("click", () => setMode("build"));
el("mode-results").addEventListener("click", () => setMode("results"));
