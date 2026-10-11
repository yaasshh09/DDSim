{
  const box = el("contact-card");
  el("contact-open").addEventListener("click", () => box.showModal());
  box.addEventListener("click", (e) => { if (e.target === box) box.close(); });
}
