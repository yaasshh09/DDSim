"use strict";

const THEME_KEY = "ddsim-theme";
const OS_DARK = matchMedia("(prefers-color-scheme: dark)");

function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch (problem) {
    return null;
  }
}

function useTheme(name) {
  document.documentElement.dataset.theme = name;
  const button = document.getElementById("theme");
  if (button) button.textContent = name === "dark" ? "light mode" : "dark mode";
  if (typeof retheme === "function") retheme();
}

useTheme(storedTheme() || (OS_DARK.matches ? "dark" : "light"));

OS_DARK.addEventListener("change", () => {
  if (!storedTheme()) useTheme(OS_DARK.matches ? "dark" : "light");
});

document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("theme");
  if (!button) return;
  useTheme(document.documentElement.dataset.theme);
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch (problem) {}
    useTheme(next);
  });
});
