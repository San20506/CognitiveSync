'use strict';

const TOKEN_KEY = 'cs_token';
const ROLE_KEY  = 'cs_role';

// ── Bootstrap ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const token = localStorage.getItem(TOKEN_KEY);
  const role  = localStorage.getItem(ROLE_KEY);
  if (token && role) {
    loadDashboard(role);
  }
});

// ── Auth ────────────────────────────────────────────────────────
async function signIn() {
  const role = document.getElementById('role-select').value;
  try {
    const res = await fetch(`/api/v1/demo/token?role=${role}`);
    if (!res.ok) throw new Error(`Token request failed: ${res.status}`);
    const data = await res.json();
    localStorage.setItem(TOKEN_KEY, data.access_token);
    localStorage.setItem(ROLE_KEY, role);
    loadDashboard(role);
  } catch (err) {
    alert(`Sign in failed: ${err.message}`);
  }
}

function signOut() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ROLE_KEY);
  document.getElementById('app-view').style.display  = 'none';
  document.getElementById('login-view').style.display = 'flex';
}

// ── API helper ──────────────────────────────────────────────────
async function apiFetch(path) {
  const token = localStorage.getItem(TOKEN_KEY);
  const res = await fetch(path, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (res.status === 401) {
    signOut();
    throw new Error('Session expired');
  }
  if (!res.ok) throw new Error(`API error ${res.status} on ${path}`);
  return res.json();
}

// ── Load dashboard ──────────────────────────────────────────────
async function loadDashboard(role) {
  document.getElementById('login-view').style.display = 'none';
  document.getElementById('app-view').style.display   = 'block';

  const badge = document.getElementById('role-badge');
  badge.textContent = role.replace('_', ' ');

  const isHR = role === 'hr_admin' || role === 'hr_analyst';

  // Show/hide panels based on role
  document.getElementById('employee-panel').style.display = isHR ? 'block' : 'none';
  document.getElementById('cascade-panel').style.display  = isHR ? 'block' : 'none';

  if (isHR) {
    const [scores, teams, cascade] = await Promise.allSettled([
      apiFetch('/api/v1/scores'),
      apiFetch('/api/v1/scores/team-summary'),
      apiFetch('/api/v1/cascade-map'),
    ]);
    renderEmployeeTable(scores.status === 'fulfilled' ? scores.value : []);
    renderTeamCards(teams.status === 'fulfilled' ? teams.value : []);
    renderCascadePanel(cascade.status === 'fulfilled' ? cascade.value : null);
  } else {
    const teams = await apiFetch('/api/v1/scores/team-summary').catch(() => []);
    renderTeamCards(teams);
  }
}

// ── Helpers ─────────────────────────────────────────────────────
function riskLevel(score) {
  if (score >= 0.7) return 'HIGH';
  if (score >= 0.4) return 'MEDIUM';
  return 'LOW';
}

function riskClass(level) {
  return { HIGH: 'high', MEDIUM: 'medium', LOW: 'low' }[level] ?? 'low';
}

function shortId(uuid) {
  return uuid ? String(uuid).slice(0, 8) + '…' : '—';
}

function fmtScore(n) {
  return typeof n === 'number' ? n.toFixed(2) : '—';
}

// ── Employee table ──────────────────────────────────────────────
function renderEmployeeTable(scores) {
  const tbody = document.getElementById('employee-tbody');
  if (!scores.length) {
    tbody.innerHTML = '<tr><td colspan="6"><p class="state-msg">No score data available</p></td></tr>';
    return;
  }

  tbody.innerHTML = scores.map(s => {
    const level = riskLevel(s.burnout_score ?? 0);
    const cls   = riskClass(level);
    const fillPct = Math.round((s.burnout_score ?? 0) * 100);
    const fillColor = { high: 'var(--risk-high)', medium: 'var(--risk-medium)', low: 'var(--risk-low)' }[cls];
    const signals   = topSignals(s.top_features);
    const ci = (s.confidence_low != null && s.confidence_high != null)
      ? `[${fmtScore(s.confidence_low)}, ${fmtScore(s.confidence_high)}]`
      : '—';

    return `<tr>
      <td><code style="font-size:.8rem">${shortId(s.pseudo_id)}</code></td>
      <td><span class="badge badge-${cls}">${level}</span></td>
      <td>
        <div class="score-bar-wrap">
          <span>${fmtScore(s.burnout_score)}</span>
          <div class="score-bar"><div class="score-fill" style="width:${fillPct}%;background:${fillColor}"></div></div>
        </div>
      </td>
      <td style="color:var(--muted);font-size:.82rem">${ci}</td>
      <td>${fmtScore(s.cascade_risk)}</td>
      <td><div class="signals">${signals}</div></td>
    </tr>`;
  }).join('');
}

function topSignals(features) {
  if (!features || !Object.keys(features).length) return '<span style="color:var(--muted);font-size:.8rem">—</span>';
  return Object.entries(features)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([k]) => `<span class="signal-pill">${k.replace(/_/g, ' ')}</span>`)
    .join('');
}

// ── Team cards ──────────────────────────────────────────────────
function renderTeamCards(teams) {
  const grid = document.getElementById('team-cards');
  if (!teams.length) {
    grid.innerHTML = '<p class="state-msg">No team data available</p>';
    return;
  }

  grid.innerHTML = teams.map(t => {
    const avg   = t.avg_score ?? t.average_score ?? null;
    const level = avg != null ? riskLevel(avg) : null;
    const cls   = level ? riskClass(level) : 'low';
    const members = t.member_count ?? t.employee_count ?? '—';

    return `<div class="team-card">
      <div class="team-id">Team ${shortId(t.team_id)}</div>
      <div class="team-score risk-${cls}">${avg != null ? avg.toFixed(2) : '—'}</div>
      <div class="team-meta">${members} member${members !== 1 ? 's' : ''} · ${level ?? '—'} risk</div>
    </div>`;
  }).join('');
}

// ── Cascade panel ────────────────────────────────────────────────
function renderCascadePanel(cascadeMap) {
  const list = document.getElementById('cascade-list');
  if (!cascadeMap) {
    list.innerHTML = '<li><p class="state-msg">No cascade data available</p></li>';
    return;
  }

  // cascadeMap may be {nodes, edges} or similar shape — handle both
  const edges = cascadeMap.edges ?? cascadeMap.propagation_edges ?? [];
  const nodes = cascadeMap.nodes ?? {};

  if (!edges.length) {
    list.innerHTML = '<li><p class="state-msg">No active cascade propagation paths</p></li>';
    return;
  }

  list.innerHTML = edges.slice(0, 20).map(e => {
    const src  = shortId(e.source ?? e.from_id);
    const tgt  = shortId(e.target ?? e.to_id);
    const weight = e.weight ?? e.cascade_weight;
    const wStr = weight != null ? ` (weight: ${Number(weight).toFixed(2)})` : '';
    return `<li>
      <span class="badge badge-medium">→</span>
      <code style="font-size:.8rem">${src}</code>
      <span style="color:var(--muted)">→</span>
      <code style="font-size:.8rem">${tgt}</code>
      <span style="color:var(--muted);font-size:.8rem">${wStr}</span>
    </li>`;
  }).join('');

  if (edges.length > 20) {
    list.innerHTML += `<li style="color:var(--muted);font-size:.82rem;padding:.75rem .5rem">…and ${edges.length - 20} more edges</li>`;
  }
}
