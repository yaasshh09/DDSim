function explainButton(onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "explain";
  button.textContent = "i";
  button.setAttribute("aria-label", "explain");
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    onClick();
  });
  return button;
}

function escapeText(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInto(element, markdown) {
  const maths = [];
  const lifted = markdown.replace(/\$\$[\s\S]+?\$\$|\$[^$]+?\$/g, (tex) => {
    maths.push(tex);
    return "@@MATH" + (maths.length - 1) + "@@";
  });
  element.innerHTML = marked
    .parse(lifted)
    .replace(/@@MATH(\d+)@@/g, (_, index) => escapeText(maths[Number(index)]));
  renderMathInElement(element, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

let explaining = 0;

async function explain(topicName, knob) {
  const asked = ++explaining;
  el("drawer-knob").textContent = knob
    ? knob.name + (knob.unit ? " [" + knob.unit + "]" : "") + ": " +
      knob.explanation + " Default " + String(knob.default) + "."
    : "";
  el("drawer-title").textContent = knob ? knob.name : "";
  el("drawer-plain").textContent = "";
  el("drawer-depth").textContent = "";
  el("drawer-docs").textContent = "";
  if (topicName) {
    const response = await fetch("/api/learn/" + encodeURIComponent(topicName));
    if (response.ok) {
      const topic = await response.json();
      if (asked !== explaining) return;
      el("drawer-title").textContent = topic.title;
      renderInto(el("drawer-plain"), topic.plain);
      renderInto(el("drawer-depth"), topic.depth);
      el("drawer-docs").textContent = "More in the repo: " + topic.docs.join(", ");
    }
  }
  el("drawer").classList.add("open");
}

function markExplainable(root, topics) {
  for (const target of root.querySelectorAll("[data-topic-id]")) {
    if (target.querySelector(":scope > .explain")) continue;
    const topic = topics[target.dataset.topicId];
    target.appendChild(explainButton(() => explain(topic)));
  }
}
