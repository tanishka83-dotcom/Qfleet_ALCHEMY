/**
 * QFleet Phase 4B — Landing Page Client Script
 * Fetches data from the local API and renders all sections.
 * Missing/null values display as "--".
 */

const API_BASE = "";  // same origin

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(value) {
  if (value === null || value === undefined || value === "") return "--";
  return value;
}

function fmtNum(value, decimals) {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (isNaN(n)) return "--";
  if (decimals !== undefined) {
    return n.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  }
  return n.toLocaleString("en-US");
}

function fmtDollar(value) {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (isNaN(n)) return "--";
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function fmtPct(value) {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (isNaN(n)) return "--";
  return n.toFixed(2) + "%";
}

function fmtRuntime(value) {
  if (value === null || value === undefined) return "--";
  const n = Number(value);
  if (isNaN(n)) return "--";
  return n.toFixed(3);
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchJson(endpoint) {
  try {
    const resp = await fetch(API_BASE + endpoint);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error(`Failed to fetch ${endpoint}:`, err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Section renderers
// ---------------------------------------------------------------------------

async function renderSummary() {
  const data = await fetchJson("/api/summary");
  if (!data) return;

  document.getElementById("stat-vessels").textContent = fmt(data.vessel_count);
  document.getElementById("stat-routes").textContent = fmt(data.route_count);
  document.getElementById("stat-fuels").textContent = fmt(data.fuel_count);
  document.getElementById("stat-scenarios").textContent = fmt(data.scenario_count);
  document.getElementById("stat-opt-runs").textContent = fmt(data.optimization_run_count);
  document.getElementById("stat-todo-count").textContent = fmt(data.todo_verify_count);

  document.getElementById("fuel-names").textContent =
    data.fuel_names ? data.fuel_names.join(", ") : "--";
  document.getElementById("vessel-classes").textContent =
    data.vessel_classes ? data.vessel_classes.join(", ") : "--";

  document.getElementById("lim-todo-count").textContent = fmt(data.todo_verify_count);
}

async function renderBenchmark() {
  const data = await fetchJson("/api/benchmark");
  if (!data) return;

  const tbody = document.getElementById("benchmark-tbody");
  tbody.innerHTML = "";

  for (const row of data.benchmark_rows) {
    const tr = document.createElement("tr");
    const nDisplay = (row.n_feasible !== null && row.n_runs !== null) ? `${row.n_feasible}/${row.n_runs}` : (row.n_feasible !== null ? String(row.n_feasible) : "--");

    if (row.is_failed) {
      tr.className = "row-failed";
      tr.innerHTML =
        `<td><strong>${escapeHtml(row.instance)}</strong></td>` +
        `<td>${escapeHtml(row.algorithm)}</td>` +
        `<td class="failed-message" colspan="5" style="color: #f43f5e; font-style: italic;">Failed: no feasible plan</td>` +
        `<td>${escapeHtml(row.reference_type)}</td>` +
        `<td>${fmtPct(row.feasibility_rate_pct)}</td>` +
        `<td>${nDisplay}</td>` +
        `<td>${fmtRuntime(row.mean_runtime_s)}</td>`;
    } else {
      tr.innerHTML =
        `<td><strong>${escapeHtml(row.instance)}</strong></td>` +
        `<td>${escapeHtml(row.algorithm)}</td>` +
        `<td>${fmtDollar(row.mean_objective)}</td>` +
        `<td>${fmtDollar(row.std_objective)}</td>` +
        `<td>${fmtDollar(row.best_objective)}</td>` +
        `<td>${fmtDollar(row.worst_objective)}</td>` +
        `<td>${fmtPct(row.gap_to_ref_pct)}</td>` +
        `<td>${escapeHtml(row.reference_type)}</td>` +
        `<td>${fmtPct(row.feasibility_rate_pct)}</td>` +
        `<td>${nDisplay}</td>` +
        `<td>${fmtRuntime(row.mean_runtime_s)}</td>`;
    }

    tbody.appendChild(tr);
  }

  // Statistical tests
  const statTbody = document.getElementById("stats-tbody");
  statTbody.innerHTML = "";

  for (const test of data.statistical_tests) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${escapeHtml(test.instance)}</td>` +
      `<td>${escapeHtml(test.comparison)}</td>` +
      `<td>${fmtDollar(test.mean_diff)}</td>` +
      `<td>${test.wilcoxon_p_holm !== null ? Number(test.wilcoxon_p_holm).toExponential(3) : "--"}</td>` +
      `<td>${test.is_significant_005 ? "Yes" : "No"}</td>`;
    statTbody.appendChild(tr);
  }
}

let cachedRoutesData = [];

async function renderRouteFuelComparison() {
  const data = await fetchJson("/api/routes_fuel_comparison");
  if (!data || !data.routes) return;

  cachedRoutesData = data.routes;
  const select = document.getElementById("route-select");
  select.innerHTML = "";

  cachedRoutesData.forEach((r, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${r.route_name} (${r.distance_nm.toLocaleString()} nm, ${r.origin_dest})`;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => {
    updateRouteFuelTable(Number(e.target.value));
  });

  if (cachedRoutesData.length > 0) {
    updateRouteFuelTable(0);
  }
}

function updateRouteFuelTable(routeIndex) {
  const tbody = document.getElementById("route-fuel-tbody");
  tbody.innerHTML = "";
  const route = cachedRoutesData[routeIndex];
  if (!route || !route.fuels) return;

  route.fuels.forEach(f => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td><strong>${escapeHtml(f.fuel_name)}</strong></td>` +
      `<td>${fmtNum(f.fuel_mass_t, 1)}</td>` +
      `<td>${fmtDollar(f.fuel_cost_usd)}</td>` +
      `<td>${fmtNum(f.co2_wtw_t, 1)}</td>` +
      `<td>${fmtDollar(f.carbon_cost_usd)}</td>` +
      `<td><strong>${fmtDollar(f.total_cost_usd)}</strong></td>`;
    tbody.appendChild(tr);
  });
}

async function renderTodoVerify() {
  const data = await fetchJson("/api/todo_verify");
  if (!data) return;

  const container = document.getElementById("todo-verify-list");
  container.innerHTML = "";

  if (data.items.length === 0) {
    container.innerHTML = "<p>No TODO_VERIFY items found.</p>";
    return;
  }

  for (const item of data.items) {
    const div = document.createElement("div");
    div.className = "todo-item";

    const match = item.match(/^(TV-\d+)\s+/);
    if (match) {
      div.innerHTML = `<span class="todo-id">${escapeHtml(match[1])}</span> ${escapeHtml(item.substring(match[0].length))}`;
    } else {
      div.textContent = item;
    }

    container.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  renderSummary();
  renderBenchmark();
  renderRouteFuelComparison();
  renderTodoVerify();
});
