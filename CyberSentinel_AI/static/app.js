/**
 * CyberSentinel RL Lab - Frontend Interactive Controller
 * Connects UI modules to Flask REST APIs for Q-Learning, SARSA, and Model-Based RL.
 */

// Global State & Chart Instances
let trainRewardChart = null;
let trainEpsilonChart = null;
let benchmarkChart = null;
let activeAlgorithm = 'Q-Learning';

const $ = (id) => document.getElementById(id);

/**
 * Escape HTML to prevent XSS in table outputs
 */
function esc(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/[&<>"']/g, (m) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[m]));
}

/**
 * Tab Navigation
 */
function switchTab(tabId) {
  // Update sidebar active state
  const tabs = ['overview', 'simulation', 'training', 'policy', 'benchmark', 'learning'];
  tabs.forEach((t) => {
    const nav = $(`nav-${t}`);
    const pane = $(`pane-${t}`);
    if (nav) nav.classList.remove('active');
    if (pane) pane.classList.remove('active');
  });

  const activeNav = $(`nav-${tabId}`);
  const activePane = $(`pane-${tabId}`);
  if (activeNav) activeNav.classList.add('active');
  if (activePane) activePane.classList.add('active');

  // Trigger data refreshes on tab entry
  if (tabId === 'overview') loadOverview();
  if (tabId === 'simulation') loadSimState();
  if (tabId === 'policy') refreshPolicyView();
  if (tabId === 'training') loadTrainHistory();
}

/**
 * Load Overview Statistics
 */
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Failed to load telemetry');

    activeAlgorithm = d.active_algorithm;
    $('topAlgoName').textContent = d.active_algorithm;
    $('sideEngineMeta').textContent = `${d.active_algorithm} · Discrete MDP`;
    $('ovActiveAlgoDesc').textContent = `Algorithm: ${d.active_algorithm}`;

    $('ovTotalEpisodes').textContent = (d.total_episodes_trained || 0).toLocaleString();
    $('ovMovingAvg').textContent = d.moving_average_reward !== undefined ? d.moving_average_reward : '—';
    $('ovSuccessRate').innerHTML = `${d.task_success_rate || 0}<i>%</i>`;
    $('ovMissedThreatRate').innerHTML = `${d.missed_threat_rate || 0}<i>%</i>`;
    $('ovEpsilon').textContent = d.current_exploration_rate !== undefined ? d.current_exploration_rate : '—';
    $('ovEvalReward').textContent = d.average_eval_reward !== undefined ? d.average_eval_reward : '—';
    $('ovUnnecessaryBlock').textContent = `${d.unnecessary_block_rate || 0}%`;

  } catch (err) {
    console.error('Error loading overview:', err);
  }
}

/**
 * Load Simulation Environment State
 */
async function loadSimState() {
  try {
    const res = await fetch('/api/env/state');
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Failed to fetch state');

    updateSimUI(d);
  } catch (err) {
    console.error('Error fetching sim state:', err);
  }
}

/**
 * Update Simulation UI with State Data
 */
function updateSimUI(d) {
  const dec = d.decoded || {};
  $('simStateBadge').textContent = `State ${d.state_id}`;
  $('simStateTitle').textContent = dec.description || `State ${d.state_id}`;

  const catBadge = $('simCategoryBadge');
  catBadge.textContent = `● ${dec.category || 'Normal'}`;
  catBadge.className = 'badge';
  if (dec.category === 'Benign') catBadge.classList.add('normal');
  else if (dec.category === 'Severe Attack') catBadge.classList.add('suspicious');
  else catBadge.classList.add('normal');

  $('simPillTraffic').textContent = `Traffic: ${dec.traffic_name || 'Low'}`;
  $('simPillLogins').textContent = `Logins: ${dec.failed_logins_name || 'Normal'}`;
  $('simPillBehavior').textContent = `Behavior: ${dec.behavior_name || 'Normal'}`;

  $('simStepCount').textContent = `${d.step} / ${d.max_steps || 20}`;
  $('simStatus').textContent = d.done ? 'Terminated' : 'Active';
  $('simStatus').style.color = d.done ? 'var(--amber)' : 'var(--green)';

  // Update Q-values display if policy available
  fetchPolicyForState(d.state_id);
}

/**
 * Fetch Q-Values for current state
 */
async function fetchPolicyForState(stateId) {
  try {
    const res = await fetch('/api/agent/policy');
    const d = await res.json();
    if (!res.ok) return;

    const p = d.policy[stateId];
    if (p && p.q_values) {
      $('qVal0').textContent = `Q: ${p.q_values[0].toFixed(2)}`;
      $('qVal1').textContent = `Q: ${p.q_values[1].toFixed(2)}`;
      $('qVal2').textContent = `Q: ${p.q_values[2].toFixed(2)}`;
    }
  } catch (e) {
    // Silent catch
  }
}

/**
 * Execute One Step in Simulation
 */
async function stepSimulation(action = null) {
  try {
    const payload = action !== null ? { action: action } : {};
    const res = await fetch('/api/env/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Step failed');

    // Update outcome display
    $('simOutcomeSub').textContent = `STEP ${d.step} OUTCOME · ${d.action_name}`;
    $('simOutcomeTitle').textContent = d.outcome;
    $('simExplanation').textContent = d.explanation;

    const rewTag = $('simRewardTag');
    rewTag.textContent = `Reward: ${d.reward > 0 ? '+' : ''}${d.reward.toFixed(1)}`;
    if (d.reward >= 0) {
      rewTag.className = 'reward-tag';
    } else {
      rewTag.className = 'reward-tag negative';
    }

    // Append to trajectory table
    appendTrajectoryRow(d);

    // Refresh state display
    loadSimState();

  } catch (err) {
    alert(`Simulation error: ${err.message}`);
  }
}

/**
 * Append Row to Trajectory Table
 */
function appendTrajectoryRow(d) {
  const tbody = $('simTrajectoryBody');
  if (tbody.querySelector('.empty')) {
    tbody.innerHTML = '';
  }

  const tr = document.createElement('tr');
  const rewColor = d.reward >= 0 ? 'var(--green)' : 'var(--red)';
  const rewSign = d.reward > 0 ? '+' : '';

  tr.innerHTML = `
    <td><b>#${d.step}</b></td>
    <td><span class="ftag">S${d.state_before}</span> ${esc(d.state_before_info.category)}</td>
    <td><span class="badge ${d.action === 2 ? 'suspicious' : 'normal'}">${d.action_name}</span></td>
    <td><b style="color: ${rewColor}">${rewSign}${d.reward.toFixed(1)}</b></td>
    <td><span class="ftag">S${d.next_state}</span> ${esc(d.next_state_info.category)}</td>
    <td>${esc(d.outcome)}</td>
  `;

  tbody.insertBefore(tr, tbody.firstChild);
}

/**
 * Reset Simulation
 */
async function resetSimulation() {
  try {
    const res = await fetch('/api/env/reset', { method: 'POST' });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Reset failed');

    $('simTrajectoryBody').innerHTML = `<tr><td colspan="6" class="empty">Simulation reset. Ready for step 1.</td></tr>`;
    $('simOutcomeTitle').textContent = 'Environment Reset';
    $('simExplanation').textContent = 'Initial connection state initialized. Ready for agent or manual action.';
    $('simRewardTag').textContent = 'Reward: +0.0';
    $('simRewardTag').className = 'reward-tag';

    loadSimState();
  } catch (err) {
    alert(`Reset error: ${err.message}`);
  }
}

/**
 * Run Entire Simulation Episode
 */
async function runEpisodeSimulation() {
  try {
    const res = await fetch('/api/env/run_episode', { method: 'POST' });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Episode execution failed');

    const tbody = $('simTrajectoryBody');
    tbody.innerHTML = d.trajectory.map((step) => {
      const rewColor = step.reward >= 0 ? 'var(--green)' : 'var(--red)';
      const rewSign = step.reward > 0 ? '+' : '';
      return `<tr>
        <td><b>#${step.step}</b></td>
        <td><span class="ftag">S${step.state}</span></td>
        <td><span class="badge ${step.action === 2 ? 'suspicious' : 'normal'}">${step.action_name}</span></td>
        <td><b style="color: ${rewColor}">${rewSign}${step.reward.toFixed(1)}</b></td>
        <td><span class="ftag">S${step.next_state}</span></td>
        <td>${esc(step.outcome)}</td>
      </tr>`;
    }).join('');

    $('simOutcomeTitle').textContent = `Episode Complete (${d.algorithm})`;
    $('simExplanation').textContent = `Simulated ${d.episode_length} sequential connection steps. Cumulative return: ${d.total_reward}`;
    $('simRewardTag').textContent = `Total Return: ${d.total_reward}`;
    $('simRewardTag').className = d.total_reward >= 0 ? 'reward-tag' : 'reward-tag negative';

    loadSimState();
  } catch (err) {
    alert(`Episode error: ${err.message}`);
  }
}

function quickRunDemo() {
  switchTab('simulation');
  runEpisodeSimulation();
}

/**
 * Training Hyperparameter Visibility Logic
 */
function onTrainAlgoChange() {
  const algo = $('trainAlgoSelect').value;
  const planningField = $('fieldPlanning');
  const alphaField = $('fieldAlpha');
  const algoHelp = $('trainAlgoHelp');

  if (algo === 'Model-Based RL') {
    planningField.style.display = 'flex';
    alphaField.style.display = 'none';
    algoHelp.textContent = 'Learns empirical transition dynamics P(s\'|s,a) & reward model with Value Iteration planning.';
  } else if (algo === 'SARSA') {
    planningField.style.display = 'none';
    alphaField.style.display = 'flex';
    algoHelp.textContent = 'On-policy TD update using the actual action selected by the behavior policy.';
  } else {
    planningField.style.display = 'none';
    alphaField.style.display = 'flex';
    algoHelp.textContent = 'Off-policy TD update using max future Q-value regardless of exploration.';
  }
}

/**
 * Start Training Routine
 */
async function startTraining() {
  const btn = $('trainBtn');
  const spinner = $('trainSpinner');
  const statusMsg = $('trainStatusMsg');

  btn.disabled = true;
  spinner.classList.remove('hidden');
  statusMsg.textContent = 'Training agent across simulated episodes…';

  const payload = {
    algorithm: $('trainAlgoSelect').value,
    episodes: parseInt($('trainEpisodes').value, 10),
    alpha: parseFloat($('trainAlpha').value),
    gamma: parseFloat($('trainGamma').value),
    epsilon: parseFloat($('trainEpsilon').value),
    planning_iterations: parseInt($('trainPlanning').value, 10),
    seed: 42
  };

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Training failed');

    statusMsg.textContent = `Completed ${d.summary.episodes_trained} episodes! Final rolling avg: ${d.summary.final_moving_avg}`;

    // Render training curves
    renderTrainingCharts(d.history);
    // Refresh policy & overview
    loadOverview();
    refreshPolicyView();

  } catch (err) {
    statusMsg.textContent = `Error: ${err.message}`;
    alert(`Training error: ${err.message}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

/**
 * Load Existing Training History
 */
async function loadTrainHistory() {
  try {
    const res = await fetch('/api/train/history');
    const d = await res.json();
    if (res.ok && d.history && d.history.episode_rewards && d.history.episode_rewards.length > 0) {
      renderTrainingCharts(d.history);
    }
  } catch (e) {
    // Silent
  }
}

/**
 * Render Training Charts with Chart.js
 */
function renderTrainingCharts(hist) {
  if (typeof Chart === 'undefined') return;

  const epRewards = hist.episode_rewards || [];
  const movingAvg = hist.moving_avg_rewards || [];
  const epsilons = hist.epsilon_history || [];
  const lengths = hist.episode_lengths || [];

  const labels = epRewards.map((_, i) => i + 1);

  // 1. Reward Chart
  const canvasReward = $('trainRewardChart');
  if (canvasReward) {
    if (trainRewardChart) trainRewardChart.destroy();
    trainRewardChart = new Chart(canvasReward, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Episode Reward',
            data: epRewards,
            borderColor: 'rgba(0, 230, 118, 0.35)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0
          },
          {
            label: '10-Episode Rolling Avg',
            data: movingAvg,
            borderColor: '#00f0ff',
            backgroundColor: 'rgba(0, 240, 255, 0.08)',
            borderWidth: 2.2,
            pointRadius: 0,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#131e2c' }, ticks: { color: '#7a8e9e', maxTicksLimit: 10 } },
          y: { grid: { color: '#131e2c' }, ticks: { color: '#7a8e9e' } }
        },
        plugins: {
          legend: { labels: { color: '#e6f0fa', boxWidth: 10 } }
        }
      }
    });
  }

  // 2. Epsilon & Length Chart
  const canvasEps = $('trainEpsilonChart');
  if (canvasEps) {
    if (trainEpsilonChart) trainEpsilonChart.destroy();
    trainEpsilonChart = new Chart(canvasEps, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Exploration Rate (ε)',
            data: epsilons,
            borderColor: '#f59e0b',
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            label: 'Episode Length (Steps)',
            data: lengths,
            borderColor: '#a855f7',
            borderWidth: 1.5,
            pointRadius: 0,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#131e2c' }, ticks: { color: '#7a8e9e', maxTicksLimit: 10 } },
          y: {
            type: 'linear',
            position: 'left',
            min: 0,
            max: 1,
            grid: { color: '#131e2c' },
            ticks: { color: '#f59e0b' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            grid: { display: false },
            ticks: { color: '#a855f7' }
          }
        },
        plugins: {
          legend: { labels: { color: '#e6f0fa', boxWidth: 10 } }
        }
      }
    });
  }
}

/**
 * Reset Active Agent
 */
async function resetActiveAgent() {
  if (!confirm('Are you sure you want to reset the active agent to an untrained state?')) return;
  try {
    const res = await fetch('/api/agent/reset', { method: 'POST' });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Reset failed');

    alert('Agent reset successfully.');
    loadOverview();
    refreshPolicyView();
  } catch (err) {
    alert(`Reset error: ${err.message}`);
  }
}

/**
 * Evaluate Active Agent on Separate Episodes
 */
async function evaluateActiveAgent() {
  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_episodes: 50, seed: 1234 })
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Evaluation failed');

    alert(
      `Evaluation Report (${d.agent}):\n` +
      `------------------------------------\n` +
      `Average Cumulative Reward: ${d.average_reward} (±${d.reward_std})\n` +
      `Task Success Rate: ${d.success_rate_percent}%\n` +
      `Missed Severe Threats: ${d.missed_threat_rate_percent}%\n` +
      `Unnecessary Blocks: ${d.unnecessary_block_rate_percent}%\n` +
      `Avg Episode Length: ${d.average_episode_length} steps`
    );

    loadOverview();
  } catch (err) {
    alert(`Evaluation error: ${err.message}`);
  }
}

/**
 * Policy Explorer: Fetch and Render Q-Table and Policies
 */
async function refreshPolicyView() {
  try {
    const res = await fetch('/api/agent/policy');
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Failed to fetch policy');

    const tbody = $('policyTableBody');
    const pol = d.policy;

    tbody.innerHTML = Object.keys(pol).map((stateId) => {
      const row = pol[stateId];
      const q = row.q_values || [0, 0, 0];
      const best = row.recommended_action;
      const actClass = `act-${best}`;

      return `<tr>
        <td><b>State ${row.state_id}</b></td>
        <td>${esc(row.description)}</td>
        <td><span class="ftag">${esc(row.category)}</span></td>
        <td style="${best === 0 ? 'color: var(--green); font-weight: 700;' : ''}">${q[0].toFixed(2)}</td>
        <td style="${best === 1 ? 'color: var(--cyan); font-weight: 700;' : ''}">${q[1].toFixed(2)}</td>
        <td style="${best === 2 ? 'color: var(--red); font-weight: 700;' : ''}">${q[2].toFixed(2)}</td>
        <td><span class="policy-badge ${actClass}">${esc(row.recommended_action_name)}</span></td>
        <td>${row.visit_count || row.total_visits || 0}</td>
      </tr>`;
    }).join('');

    // If Model-Based RL, display transition matrix section
    const transPanel = $('modelBasedTransitionsPanel');
    if (d.algorithm === 'Model-Based RL') {
      transPanel.style.display = 'block';
      loadTransitionMatrix();
    } else {
      transPanel.style.display = 'none';
    }

  } catch (err) {
    console.error('Policy view error:', err);
  }
}

/**
 * Load Model-Based Transition Matrix P(s' | s, a)
 */
async function loadTransitionMatrix() {
  const act = $('matrixActionSelect').value;
  try {
    const res = await fetch(`/api/agent/transitions?action=${act}`);
    const d = await res.json();
    if (!res.ok) return;

    const head = $('transMatrixHead');
    const body = $('transMatrixBody');

    let headHtml = '<tr><th>State</th>';
    for (let j = 0; j < 12; j++) {
      headHtml += `<th>S${j}</th>`;
    }
    headHtml += '</tr>';
    head.innerHTML = headHtml;

    body.innerHTML = d.matrix.map((row, i) => {
      let rHtml = `<tr><td><b>S${i}</b></td>`;
      row.forEach((p) => {
        const bg = p > 0.05 ? `rgba(0, 240, 255, ${Math.min(0.6, p)})` : 'transparent';
        rHtml += `<td style="background: ${bg}; font-size: 10px;">${p > 0 ? p.toFixed(2) : '0'}</td>`;
      });
      rHtml += '</tr>';
      return rHtml;
    }).join('');

  } catch (e) {
    console.error(e);
  }
}

/**
 * Benchmark Lab: Run Head-to-Head Comparison
 */
async function runBenchmark() {
  const btn = $('benchmarkBtn');
  const spinner = $('benchSpinner');
  const tbody = $('benchmarkTableBody');

  btn.disabled = true;
  spinner.classList.remove('hidden');
  tbody.innerHTML = `<tr><td colspan="8" class="empty">Benchmarking Q-Learning, SARSA, Model-Based RL, and Rule Baseline on identical evaluation episodes…</td></tr>`;

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ train_episodes: 300, eval_episodes: 40, seed: 42 })
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Benchmark comparison failed');

    const results = d.benchmark_results;

    tbody.innerHTML = results.map((r) => {
      return `<tr>
        <td><b>${esc(r.agent)}</b></td>
        <td><span class="ftag">${esc(r.paradigm)}</span></td>
        <td><b style="color: var(--cyan);">${r.average_reward}</b></td>
        <td>&plusmn;${r.reward_std}</td>
        <td><b style="color: var(--green);">${r.success_rate_percent}%</b></td>
        <td>${r.missed_threat_rate_percent}%</td>
        <td>${r.unnecessary_block_rate_percent}%</td>
        <td>${r.training_interactions.toLocaleString()}</td>
      </tr>`;
    }).join('');

    renderBenchmarkChart(results);

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty" style="color: var(--red);">Benchmark error: ${esc(err.message)}</td></tr>`;
    alert(`Benchmark error: ${err.message}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
  }
}

/**
 * Render Benchmark Comparison Bar Chart
 */
function renderBenchmarkChart(results) {
  if (typeof Chart === 'undefined') return;

  const canvas = $('benchmarkChart');
  if (!canvas) return;

  if (benchmarkChart) benchmarkChart.destroy();

  const labels = results.map((r) => r.agent);
  const rewards = results.map((r) => r.average_reward);
  const successRates = results.map((r) => r.success_rate_percent);

  benchmarkChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Average Cumulative Return',
          data: rewards,
          backgroundColor: '#00f0ff',
          borderRadius: 4
        },
        {
          label: 'Task Success Rate (%)',
          data: successRates,
          backgroundColor: '#00e676',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { color: '#e6f0fa', font: { weight: 600 } } },
        y: { grid: { color: '#131e2c' }, ticks: { color: '#7a8e9e' } }
      },
      plugins: {
        legend: { labels: { color: '#e6f0fa', boxWidth: 10 } }
      }
    }
  });
}

/**
 * Global Initialization on DOM Ready
 */
document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
  loadSimState();
  refreshPolicyView();
  onTrainAlgoChange();
});
