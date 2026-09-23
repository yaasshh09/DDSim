function exact(v) {
  const size = Math.abs(v);
  return size !== 0 && (size >= 1e4 || size < 1e-3) ? v.toExponential() : String(v);
}

function regionBox(field, value, title) {
  const box = document.createElement("input");
  box.type = "text";
  box.value = exact(value);
  box.title = title;
  box.dataset.region = field;
  return box;
}

function regionRow(region) {
  const row = document.createElement("div");
  row.className = "region";
  const dopant = document.createElement("select");
  for (const type of ["p", "n"]) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    dopant.appendChild(option);
  }
  dopant.value = region.dopant;
  dopant.dataset.region = "dopant";
  dopant.title = "dopant: p for acceptors, n for donors";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "flat";
  remove.textContent = "x";
  remove.title = "remove this region";
  remove.dataset.region = "remove";
  remove.addEventListener("click", () => row.remove());
  row.append(
    dopant,
    regionBox("length", region.length, "length [cm]"),
    regionBox("concentration", region.concentration, "doping [cm^-3]"),
    remove
  );
  return row;
}

const STACK_NOTE =
  "A device you built. The solver is validated against DEVSIM; this " +
  "structure is not validated by anything, so its numbers are the validated " +
  "solver's answer on an unchecked structure.";

function showRegions(regions) {
  el("stack-note").textContent = STACK_NOTE;
  el("stack").hidden = !regions;
  el("regions").textContent = "";
  for (const region of regions || []) el("regions").appendChild(regionRow(region));
}

function addRegion() {
  const rows = el("regions").children;
  const last = rows.length ? collectRegions().pop() : null;
  el("regions").appendChild(regionRow({
    dopant: last && last.dopant === "p" ? "n" : "p",
    length: last ? last.length : 5e-5,
    concentration: last ? last.concentration : 1e16,
  }));
}

function collectRegions() {
  if (el("stack").hidden) return null;
  return Array.from(el("regions").children, (row, index) => {
    const region = { dopant: row.querySelector("[data-region=dopant]").value };
    for (const field of ["length", "concentration"]) {
      const text = row.querySelector("[data-region=" + field + "]").value;
      const value = Number(text);
      if (text.trim() === "" || !isFinite(value)) {
        throw new Error("region " + (index + 1) + ": " + field + " is not a number: " + text);
      }
      region[field] = value;
    }
    return region;
  });
}

function saveDevice() {
  let device;
  try {
    device = deviceRequest();
  } catch (problem) {
    el("message").textContent = String(problem.message || problem);
    return;
  }
  const file = new Blob([JSON.stringify(device, null, 2) + "\n"], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(file);
  link.download = device.kind + ".json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

async function loadDevice(file) {
  let device;
  try {
    device = JSON.parse(await file.text());
  } catch (problem) {
    el("message").textContent = file.name + " is not JSON: " + problem.message;
    return;
  }
  if (!device || !state.schema.devices[device.kind]) {
    el("message").textContent =
      file.name + " names no device this page has: " + JSON.stringify(device && device.kind);
    return;
  }
  putDevice(device);
  const known = new Set(state.schema.devices[device.kind].map((p) => p.name));
  const unknown = Object.keys(device.parameters || {}).filter(
    (name) =>
      !known.has(name) &&
      !(name === "regions" && state.schema.regions[device.kind]) &&
      !(state.schema.drawings[device.kind] && name in state.schema.drawings[device.kind])
  );
  el("message").textContent = unknown.length
    ? file.name + " sets " + unknown.join(", ") + ", which " + device.kind +
      " does not have. Left out."
    : "";
}

el("region-add").addEventListener("click", addRegion);
el("device-save").addEventListener("click", saveDevice);
el("device-open").addEventListener("click", () => el("device-load").click());
el("device-load").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (file) loadDevice(file);
});
