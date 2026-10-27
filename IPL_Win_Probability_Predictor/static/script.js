const form = document.getElementById("predict-form");
const predictButton = document.getElementById("predict-button");
const resetButton = document.getElementById("reset-button");
const errorBox = document.getElementById("form-error");
const emptyResult = document.getElementById("empty-result");
const loadingResult = document.getElementById("loading-result");
const predictionResult = document.getElementById("prediction-result");

let options = { teams: [], venues: [], max_overs: 20 };
const autocompleteConfig = [
  { id: "batting_team", source: "teams" },
  { id: "bowling_team", source: "teams" },
  { id: "venue", source: "venues" }
];
const autocompleteState = {};

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}
function hideResults() {
  emptyResult.hidden = true;
  loadingResult.hidden = true;
  predictionResult.hidden = true;
}
function normalize(value) {
  return String(value || "").toLocaleLowerCase().trim();
}
function setupAutocomplete(config) {
  const input = document.getElementById(config.id);
  const list = document.getElementById(`${config.id}-suggestions`);
  autocompleteState[config.id] = { active: -1, items: [] };

  function closeList() {
    list.hidden = true;
    autocompleteState[config.id].active = -1;
  }
  function choose(value) {
    input.value = value;
    closeList();
    input.dataset.selected = value;
    clearError();
    if (config.id === "batting_team" || config.id === "bowling_team") {
      const otherId = config.id === "batting_team" ? "bowling_team" : "batting_team";
      const other = document.getElementById(otherId);
      if (other.value === value) {
        other.value = "";
        delete other.dataset.selected;
      }
    }
  }
  function render() {
    const query = normalize(input.value);
    const all = options[config.source] || [];
    const otherTeamId = config.id === "batting_team" ? "bowling_team" : "batting_team";
    const otherTeam = config.source === "teams" ? normalize(document.getElementById(otherTeamId).value) : "";
    const matches = all.filter(item => normalize(item).includes(query) && normalize(item) !== otherTeam).slice(0, 80);
    autocompleteState[config.id].items = matches;
    autocompleteState[config.id].active = -1;
    list.replaceChildren();
    if (!query && !matches.length) { closeList(); return; }
    if (!matches.length) {
      const empty = document.createElement("div");
      empty.className = "suggestion-empty";
      empty.textContent = "No matching options";
      list.appendChild(empty);
    } else {
      matches.forEach((value, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        button.role = "option";
        button.textContent = value;
        button.addEventListener("mousedown", event => event.preventDefault());
        button.addEventListener("click", () => choose(value));
        list.appendChild(button);
      });
    }
    list.hidden = false;
  }
  function setActive(index) {
    const state = autocompleteState[config.id];
    const buttons = [...list.querySelectorAll(".suggestion")];
    if (!buttons.length) return;
    state.active = (index + buttons.length) % buttons.length;
    buttons.forEach((button, i) => {
      button.classList.toggle("active", i === state.active);
      button.setAttribute("aria-selected", String(i === state.active));
    });
    buttons[state.active].scrollIntoView({ block: "nearest" });
  }
  input.addEventListener("input", () => {
    delete input.dataset.selected;
    clearError();
    render();
  });
  input.addEventListener("focus", render);
  input.addEventListener("keydown", event => {
    const state = autocompleteState[config.id];
    if (event.key === "ArrowDown") { event.preventDefault(); if (list.hidden) render(); setActive(state.active + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); setActive(state.active <= 0 ? state.items.length - 1 : state.active - 1); }
    else if (event.key === "Enter" && !list.hidden && state.items.length) {
      event.preventDefault(); choose(state.items[state.active >= 0 ? state.active : 0]);
    } else if (event.key === "Escape") closeList();
  });
  input.addEventListener("blur", () => setTimeout(closeList, 130));
}

autocompleteConfig.forEach(setupAutocomplete);

async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load team and venue options.");
    options = data;
  } catch (error) {
    showError(`${error.message} Make sure the model has been trained.`);
  }
}
loadOptions();

function getLegalBalls(overs) {
  const value = Number(overs);
  if (!Number.isFinite(value) || value < 0) throw new Error("Enter valid overs completed.");
  const whole = Math.floor(value);
  const fractional = Math.round((value - whole) * 10);
  if (fractional < 0 || fractional > 5 || Math.abs(value - (whole + fractional / 10)) > 0.00001) {
    throw new Error("Use cricket notation, for example 14.3 means 14 overs and 3 balls.");
  }
  return whole * 6 + fractional;
}
function validateForm(data) {
  if (!options.teams.length || !options.venues.length) throw new Error("Model options are unavailable. Train the model first.");
  if (!options.teams.includes(data.batting_team)) throw new Error("Select the batting team from the autocomplete suggestions.");
  if (!options.teams.includes(data.bowling_team)) throw new Error("Select the bowling team from the autocomplete suggestions.");
  if (!options.venues.includes(data.venue)) throw new Error("Select the venue from the autocomplete suggestions.");
  if (data.batting_team === data.bowling_team) throw new Error("Batting and bowling teams must be different.");
  for (const key of ["score", "wickets_lost", "target", "overs_completed"]) {
    if (data[key] === "" || !Number.isFinite(Number(data[key]))) throw new Error("Fill in all numeric fields with valid numbers.");
  }
  data.score = Number(data.score);
  data.wickets_lost = Number(data.wickets_lost);
  data.target = Number(data.target);
  data.overs_completed = Number(data.overs_completed);
  if (!Number.isInteger(data.score) || data.score < 0) throw new Error("Current score must be a non-negative whole number.");
  if (!Number.isInteger(data.target) || data.target < 1) throw new Error("Target must be a positive whole number.");
  if (!Number.isInteger(data.wickets_lost) || data.wickets_lost < 0 || data.wickets_lost > 10) throw new Error("Wickets lost must be between 0 and 10.");
  const balls = getLegalBalls(data.overs_completed);
  if (balls > Number(options.max_overs || 20) * 6) throw new Error("Overs completed cannot exceed 20.0.");
  if (balls === 0 && data.score > 0) throw new Error("Overs must be greater than zero if runs have been scored.");
  data._balls_bowled = balls;
  return data;
}
function showPrediction(result) {
  emptyResult.hidden = true;
  loadingResult.hidden = true;
  predictionResult.hidden = false;
  document.getElementById("result-team").textContent = result.batting_team;
  document.getElementById("probability").textContent = Number(result.win_probability).toFixed(1);
  document.getElementById("summary").textContent = result.summary;
  document.getElementById("runs-remaining").textContent = result.runs_remaining;
  document.getElementById("balls-remaining").textContent = result.balls_remaining;
  document.getElementById("current-run-rate").textContent = Number(result.current_run_rate).toFixed(2);
  document.getElementById("required-run-rate").textContent = Number(result.required_run_rate).toFixed(2);
  const bar = document.getElementById("progress-bar");
  bar.style.width = "0%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    bar.style.width = `${Math.max(0, Math.min(100, Number(result.win_probability)))}%`;
  }));
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  clearError();
  try {
    const raw = Object.fromEntries(new FormData(form).entries());
    const data = validateForm(raw);
    delete data._balls_bowled;
    hideResults();
    loadingResult.hidden = false;
    predictButton.disabled = true;
    predictButton.querySelector(".button-label").textContent = "Calculating...";
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction request failed.");
    showPrediction(result);
  } catch (error) {
    loadingResult.hidden = true;
    emptyResult.hidden = false;
    showError(error.message || "Something went wrong.");
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector(".button-label").textContent = "Predict win probability";
  }
});
resetButton.addEventListener("click", () => {
  form.reset();
  autocompleteConfig.forEach(({id}) => {
    const input = document.getElementById(id);
    delete input.dataset.selected;
    document.getElementById(`${id}-suggestions`).hidden = true;
  });
  clearError();
  hideResults();
  emptyResult.hidden = false;
  document.getElementById("progress-bar").style.width = "0%";
});
