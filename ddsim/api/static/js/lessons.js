async function lessons() {
  const listed = await (await fetch("/api/lessons")).json();
  for (const entry of listed) {
    const option = document.createElement("option");
    option.value = entry.name;
    option.textContent = entry.title;
    option.title = entry.summary;
    el("lesson").appendChild(option);
  }
}

async function openLesson(name) {
  if (!name) return leaveLesson();
  const response = await fetch("/api/lessons/" + encodeURIComponent(name));
  if (!response.ok) return;
  const lesson = await response.json();

  el("lesson-title").textContent = lesson.title;
  el("lesson-summary").textContent = lesson.summary;
  const steps = el("lesson-steps");
  steps.textContent = "";
  for (const step of lesson.steps) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = step.title;
    item.appendChild(title);
    if (step.request) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "flat";
      button.textContent = "set this up";
      button.addEventListener("click", () => setUp(step.request, lesson.mesh_note));
      item.appendChild(button);
    }
    const text = document.createElement("div");
    renderInto(text, step.text);
    item.appendChild(text);
    steps.appendChild(item);
  }
  renderInto(el("lesson-look"), lesson.look_for);
  renderInto(el("lesson-saw"), lesson.explanation);
  el("lesson-claims").textContent =
    "Held by tests on this lesson's own device: " + lesson.claims.join(", ") + ".";
  el("lesson-saw").parentElement.open = false;
  el("lesson-panel").hidden = false;
  setUp(lesson.request, lesson.mesh_note);
}

function leaveLesson() {
  el("lesson-panel").hidden = true;
  el("lesson").value = "";
}

el("lesson").addEventListener("change", (event) => openLesson(event.target.value));
el("lesson-leave").addEventListener("click", leaveLesson);
lessons();
