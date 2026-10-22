/**
 * CyberSentinel AI — Intelligent Cyber Threat Detection & Simulation Platform
 * Frontend Interactive Controller
 */

// Global Application State
let trafficChartInstance = null;
let compareChartInstance = null;
let currentData = null;
let allEvents = [];
let filteredEvents = [];
let currentPage = 1;
const PAGE_SIZE = 10;

// Shorthand selector helper
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
 * Format bytes into human-readable string (e.g., 14.2 KB)
 */
function formatBytes(bytes) {
  const b = Number(bytes);
  if (isNaN(b) || b <= 0) return '0 B';
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Update UI when algorithm changes
 */
function onAlgorithmChange() {
  const algo = $('algorithm').value;
  const contField = $('contaminationField');
  const algoHelp = $('algoHelp');

  if (algo === 'Isolation Forest') {
    algoHelp.textContent = 'Unsupervised outlier isolation. Contamination slider sets anomaly threshold.';
    contField.style.opacity = '1';
    contField.style.pointerEvents = 'auto';
  } else if (algo === 'Decision Tree') {
    algoHelp.textContent = 'Supervised white-box tree. Splits features on Gini impurity to expose clear rules.';
    contField.style.opacity = '0.55';
    contField.style.pointerEvents = 'none';
  } else {
    algoHelp.textContent = 'Supervised ensemble of 100 decorrelated trees with balanced class weighting.';
    contField.style.opacity = '0.55';
    contField.style.pointerEvents = 'none';
  }
}

/**
 * Update contamination percentage label
 */
function updateContamination(val) {
  $('contLabel').textContent = `${val}%`;
}

/**
 * Display error banner with title and message
 */
function showError(title, message) {
  const errorBox = $('error');
  $('errorTitle').textContent = title || 'Analysis Notice';
  $('errorMessage').textContent = message || 'An unexpected error occurred.';
  errorBox.classList.remove('hidden');
  errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Dismiss error banner
 */
function dismissError() {
  $('error').classList.add('hidden');
}

/**
 * Handle File Selection via File Input or Drag & Drop
 */
function onFileSelected(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showError('Invalid File Type', 'Please upload a valid CSV (.csv) file.');
      input.value = '';
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      showError('File Too Large', 'Uploaded file exceeds the maximum 8 MB limit.');
      input.value = '';
      return;
    }

    dismissError();
    $('fileName').textContent = file.name;
    $('fileSize').textContent = formatBytes(file.size);
    $('dropPrompt').classList.add('hidden');
    $('fileInfo').classList.remove('hidden');
    $('datasetStatus').textContent = `Custom dataset selected: ${file.name} (${formatBytes(file.size)})`;
  }
}

/**
 * Clear custom uploaded file and revert to built-in dataset
 */
function clearUploadedFile(event) {
  if (event) event.stopPropagation();
  $('file').value = '';
  $('dropPrompt').classList.remove('hidden');
  $('fileInfo').classList.add('hidden');
  $('datasetStatus').textContent = 'Built-in sample dataset (800 records)';
  analyze();
}

/**
 * Drag and Drop Setup
 */
function setupDragAndDrop() {
  const dropZone = $('dropZone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      $('file').files = files;
      onFileSelected($('file'));
    }
  }, false);
}

/**
 * Analyze Traffic Engine Execution
 */
async function analyze() {
  dismissError();

  const analyzeBtn = $('analyzeBtn');
  const heroRunBtn = $('heroRunBtn');
  const analyzeSpinner = analyzeBtn.querySelector('.spinner');
  const heroSpinner = heroRunBtn.querySelector('.spinner');

  // Activate loading indicators
  analyzeBtn.disabled = true;
  heroRunBtn.disabled = true;
  if (analyzeSpinner) analyzeSpinner.classList.remove('hidden');
  if (heroSpinner) heroSpinner.classList.remove('hidden');
  $('datasetStatus').textContent = '● Training models and evaluating telemetry…';

  const formData = new FormData();
  formData.append('algorithm', $('algorithm').value);
  formData.append('contamination', (Number($('contamination').value) / 100).toFixed(2));

  const fileInput = $('file');
  if (fileInput.files && fileInput.files[0]) {
    formData.append('file', fileInput.files[0]);
  }

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Server returned an error during analysis.');
    }

    currentData = data;
    renderDashboard(data);

    const dsName = fileInput.files && fileInput.files[0]
      ? fileInput.files[0].name
      : 'Built-in sample dataset';
    $('datasetStatus').textContent = `● Active: ${dsName} · ${data.summary.total_events} connection flows`;

  } catch (err) {
    showError('Analysis Failure', err.message);
    $('datasetStatus').textContent = '● Analysis failed';
  } finally {
    analyzeBtn.disabled = false;
    heroRunBtn.disabled = false;
    if (analyzeSpinner) analyzeSpinner.classList.add('hidden');
    if (heroSpinner) heroSpinner.classList.add('hidden');
  }
}

/**
 * Render Complete Dashboard from Response
 */
function renderDashboard(data) {
  const sum = data.summary;
  const m = data.metrics;

  // 1. Primary Overview Telemetry Cards
  $('records').textContent = sum.total_events.toLocaleString();
  $('normalCount').textContent = sum.normal_count.toLocaleString();
  $('normalPct').textContent = `${((sum.normal_count / sum.total_events) * 100).toFixed(1)}% verified benign`;

  $('suspiciousCount').textContent = sum.suspicious_count.toLocaleString();
  $('suspiciousPct').textContent = `${((sum.suspicious_count / sum.total_events) * 100).toFixed(1)}% anomalous / attack`;

  $('detectionRate').innerHTML = `${sum.detection_rate}<i>%</i>`;
  $('detectionSub').textContent = `${sum.suspicious_count} flagged of ${sum.total_events} events`;

  // 2. Secondary ML Metrics
  $('accuracy').textContent = `${m.accuracy}%`;
  $('precision').textContent = `${m.precision}%`;
  $('recall').textContent = `${m.recall}%`;
  $('f1').textContent = `${m.f1}%`;

  // 3. Metric Progress Bars
  $('barAccVal').textContent = `${m.accuracy}%`;
  $('barAccFill').style.width = `${Math.min(100, Math.max(0, m.accuracy))}%`;

  $('barPrecVal').textContent = `${m.precision}%`;
  $('barPrecFill').style.width = `${Math.min(100, Math.max(0, m.precision))}%`;

  $('barRecVal').textContent = `${m.recall}%`;
  $('barRecFill').style.width = `${Math.min(100, Math.max(0, m.recall))}%`;

  $('barF1Val').textContent = `${m.f1}%`;
  $('barF1Fill').style.width = `${Math.min(100, Math.max(0, m.f1))}%`;

  // 4. Confusion Matrix (2x2)
  const cm = m.cm; // [[TN, FP], [FN, TP]]
  if (cm && cm.length === 2 && cm[0].length === 2) {
    $('cmTN').textContent = cm[0][0];
    $('cmFP').textContent = cm[0][1];
    $('cmFN').textContent = cm[1][0];
    $('cmTP').textContent = cm[1][1];
  }

  // 5. Traffic Profile Chart (Doughnut) with Fallback
  renderTrafficChart(sum.normal_count, sum.suspicious_count);

  // 6. Intelligent Feature Insight Text
  const highRiskNote = sum.suspicious_count > 0
    ? `Flagged flows demonstrate correlated anomalies (high packet rates, bursty bytes, or elevated failed logins).`
    : `No suspicious traffic detected based on current threshold.`;

  $('insight').textContent = `${data.algorithm} processed ${sum.total_events} flows and classified ${sum.suspicious_count} (${sum.detection_rate}%) as suspicious. ${highRiskNote} Evaluated on ${m.test_rows} held-out test records.`;
  $('eventNote').textContent = data.note || 'Classification output indicates model flags on network telemetry.';

  // 7. Store Events & Render Table
  allEvents = data.events || [];
  currentPage = 1;
  onFilterEvents();
}

/**
 * Render Traffic Distribution Doughnut Chart with Chart.js or Fallback
 */
function renderTrafficChart(normal, suspicious) {
  const total = normal + suspicious;
  const normalPct = total > 0 ? ((normal / total) * 100).toFixed(1) : '0.0';
  const suspPct = total > 0 ? ((suspicious / total) * 100).toFixed(1) : '0.0';

  $('legendNormalPct').textContent = `${normalPct}%`;
  $('legendSuspPct').textContent = `${suspPct}%`;

  const canvas = $('trafficChart');
  const fallback = $('trafficChartFallback');

  // Check if Chart.js is loaded
  if (typeof Chart === 'undefined') {
    if (canvas) canvas.classList.add('hidden');
    if (fallback) {
      fallback.classList.remove('hidden');
      $('fallbackNormalBar').style.width = `${normalPct}%`;
      $('fallbackSuspBar').style.width = `${suspPct}%`;
      $('fallbackNormalLabel').textContent = `Normal: ${normalPct}% (${normal})`;
      $('fallbackSuspLabel').textContent = `Suspicious: ${suspPct}% (${suspicious})`;
    }
    return;
  }

  if (fallback) fallback.classList.add('hidden');
  if (canvas) canvas.classList.remove('hidden');

  if (trafficChartInstance) {
    trafficChartInstance.destroy();
  }

  trafficChartInstance = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['Normal Traffic', 'Suspicious Traffic'],
      datasets: [{
        data: [normal, suspicious],
        backgroundColor: ['#00e676', '#ff3366'],
        borderColor: '#0d1522',
        borderWidth: 4,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#091320',
          borderColor: '#1b283b',
          borderWidth: 1,
          titleColor: '#e6f0fa',
          bodyColor: '#8295ab',
          callbacks: {
            label: function(ctx) {
              const val = ctx.raw || 0;
              const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
              return ` ${ctx.label}: ${val.toLocaleString()} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

/**
 * Filter and Search Event Records
 */
function onFilterEvents() {
  const query = ($('eventSearch').value || '').toLowerCase().trim();
  const protocolFilter = $('protocolFilter').value;
  const statusFilter = $('statusFilter').value;

  filteredEvents = allEvents.filter((e) => {
    // Protocol filter
    if (protocolFilter !== 'ALL' && e.protocol.toUpperCase() !== protocolFilter.toUpperCase()) {
      return false;
    }

    // Status filter
    if (statusFilter !== 'ALL' && e.prediction !== statusFilter) {
      return false;
    }

    // Query search
    if (query) {
      const matchId = e.id.toLowerCase().includes(query);
      const matchProto = e.protocol.toLowerCase().includes(query);
      const matchStatus = e.prediction.toLowerCase().includes(query);
      const matchPackets = String(e.packets).includes(query);
      const matchBytes = String(e.bytes).includes(query);
      if (!matchId && !matchProto && !matchStatus && !matchPackets && !matchBytes) {
        return false;
      }
    }

    return true;
  });

  currentPage = 1;
  renderEventPage();
}

/**
 * Render Paginated Event Table
 */
function renderEventPage() {
  const tbody = $('eventRows');
  const total = filteredEvents.length;

  if (total === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">No events match the active search and filter criteria.</td></tr>`;
    $('pageInfo').textContent = 'Showing 0 of 0 events';
    $('prevPageBtn').disabled = true;
    $('nextPageBtn').disabled = true;
    $('pageDisplay').textContent = '1';
    return;
  }

  const totalPages = Math.ceil(total / PAGE_SIZE);
  currentPage = Math.max(1, Math.min(currentPage, totalPages));

  const startIdx = (currentPage - 1) * PAGE_SIZE;
  const endIdx = Math.min(startIdx + PAGE_SIZE, total);
  const pageItems = filteredEvents.slice(startIdx, endIdx);

  tbody.innerHTML = pageItems.map((e) => {
    const isSusp = e.prediction === 'Suspicious';
    const badgeClass = isSusp ? 'suspicious' : 'normal';
    const badgeIcon = isSusp ? '⚑' : '●';
    const failedClass = e.failed_logins > 0 ? 'failed-warning' : '';

    return `<tr>
      <td><b>${esc(e.id)}</b></td>
      <td>${e.duration}s</td>
      <td><span class="ftag">${esc(e.protocol)}</span></td>
      <td>${e.packets.toLocaleString()}</td>
      <td>${formatBytes(e.bytes)}</td>
      <td class="${failedClass}">${e.failed_logins}</td>
      <td>${e.score}%</td>
      <td>
        <span class="badge ${badgeClass}">${badgeIcon} ${esc(e.prediction)}</span>
      </td>
    </tr>`;
  }).join('');

  $('pageInfo').textContent = `Showing ${startIdx + 1}–${endIdx} of ${total} events`;
  $('pageDisplay').textContent = `${currentPage} / ${totalPages}`;
  $('prevPageBtn').disabled = currentPage <= 1;
  $('nextPageBtn').disabled = currentPage >= totalPages;
}

/**
 * Handle Pagination Change
 */
function changePage(delta) {
  currentPage += delta;
  renderEventPage();
}

/**
 * Benchmark Lab: Model Comparison
 */
async function compareModels() {
  const compareBtn = $('compareBtn');
  const spinner = compareBtn.querySelector('.spinner');
  const tableBody = $('compareTableBody');

  compareBtn.disabled = true;
  if (spinner) spinner.classList.remove('hidden');
  tableBody.innerHTML = `<tr><td colspan="6" class="empty">Training Decision Tree, Random Forest, and Isolation Forest on identical test split…</td></tr>`;

  const formData = new FormData();
  const fileInput = $('file');
  if (fileInput.files && fileInput.files[0]) {
    formData.append('file', fileInput.files[0]);
  }

  try {
    const response = await fetch('/api/compare', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Comparison benchmark failed.');
    }

    renderCompareResults(data.results, data.note);

  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="6" class="empty" style="color: var(--red);">Benchmark error: ${esc(err.message)}</td></tr>`;
    showError('Comparison Error', err.message);
  } finally {
    compareBtn.disabled = false;
    if (spinner) spinner.classList.add('hidden');
  }
}

/**
 * Render Benchmark Lab Results Table & Chart
 */
function renderCompareResults(results, note) {
  const tableBody = $('compareTableBody');
  const canvas = $('compareChart');
  const fallback = $('compareChartFallback');

  $('compareNote').textContent = note || 'Benchmarks calculated on an identical 25% held-out test split.';

  // Paradigm labels
  const paradigms = {
    'Decision Tree': 'Supervised (White-Box)',
    'Random Forest': 'Supervised Ensemble',
    'Isolation Forest': 'Unsupervised Anomaly'
  };

  tableBody.innerHTML = results.map((r) => `
    <tr>
      <td><b>${esc(r.algorithm)}</b></td>
      <td><span class="ftag">${paradigms[r.algorithm] || 'ML'}</span></td>
      <td><b style="color: #00f0ff">${r.accuracy}%</b></td>
      <td>${r.precision}%</td>
      <td>${r.recall}%</td>
      <td><b style="color: #a855f7">${r.f1}%</b></td>
    </tr>
  `).join('');

  // Grouped Bar Chart
  if (typeof Chart === 'undefined') {
    if (canvas) canvas.classList.add('hidden');
    if (fallback) fallback.classList.remove('hidden');
    return;
  }

  if (fallback) fallback.classList.add('hidden');
  if (canvas) canvas.classList.remove('hidden');

  if (compareChartInstance) {
    compareChartInstance.destroy();
  }

  const labels = results.map((r) => r.algorithm);
  compareChartInstance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Accuracy',
          data: results.map((r) => r.accuracy),
          backgroundColor: '#00f0ff',
          borderRadius: 4
        },
        {
          label: 'Precision',
          data: results.map((r) => r.precision),
          backgroundColor: '#3b82f6',
          borderRadius: 4
        },
        {
          label: 'Recall',
          data: results.map((r) => r.recall),
          backgroundColor: '#ff3366',
          borderRadius: 4
        },
        {
          label: 'F1 Score',
          data: results.map((r) => r.f1),
          backgroundColor: '#a855f7',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          grid: { color: '#1b283b' },
          ticks: { color: '#8295ab' },
          title: { display: true, text: 'Score (%)', color: '#8295ab' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#e6f0fa', font: { weight: 600 } }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#e6f0fa',
            boxWidth: 10,
            font: { family: 'DM Mono', size: 10 }
          }
        },
        tooltip: {
          backgroundColor: '#091320',
          borderColor: '#1b283b',
          borderWidth: 1,
          titleColor: '#e6f0fa',
          bodyColor: '#8295ab'
        }
      }
    }
  });
}

/**
 * Reset Demo to Initial State
 */
function resetDemo() {
  dismissError();
  $('algorithm').value = 'Random Forest';
  $('contamination').value = '15';
  $('contLabel').textContent = '15%';
  onAlgorithmChange();

  // Clear file input and file preview
  $('file').value = '';
  $('dropPrompt').classList.remove('hidden');
  $('fileInfo').classList.add('hidden');
  $('datasetStatus').textContent = 'Built-in sample dataset (800 records)';

  // Reset filters
  $('eventSearch').value = '';
  $('protocolFilter').value = 'ALL';
  $('statusFilter').value = 'ALL';

  // Run analysis on clean sample dataset
  analyze();

  // Also reset benchmark table
  $('compareTableBody').innerHTML = `<tr><td colspan="6" class="empty">Click "Run Model Comparison" to benchmark algorithms.</td></tr>`;
  if (compareChartInstance) {
    compareChartInstance.destroy();
    compareChartInstance = null;
  }
}

/**
 * Preview Dataset Modal
 */
async function openPreviewModal() {
  const modal = $('previewModal');
  const head = $('previewHead');
  const body = $('previewBody');
  const meta = $('previewMeta');

  modal.classList.remove('hidden');
  meta.textContent = 'Loading dataset records…';
  head.innerHTML = '';
  body.innerHTML = `<tr><td class="empty">Fetching sample records…</td></tr>`;

  try {
    const res = await fetch('/api/preview');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to fetch preview.');

    meta.textContent = `Showing first 10 of ${data.total_records} rows in active sample dataset`;
    head.innerHTML = `<tr>${data.columns.map((c) => `<th>${esc(c.toUpperCase())}</th>`).join('')}</tr>`;
    body.innerHTML = data.rows.map((row) => `
      <tr>${data.columns.map((c) => `<td>${esc(row[c])}</td>`).join('')}</tr>
    `).join('');

  } catch (err) {
    meta.textContent = 'Could not load preview';
    body.innerHTML = `<tr><td class="empty" style="color: var(--red);">Error: ${esc(err.message)}</td></tr>`;
  }
}

function closePreviewModal() {
  $('previewModal').classList.add('hidden');
}

function closeModalOnBackdrop(e) {
  if (e.target && e.target.id === 'previewModal') {
    closePreviewModal();
  }
}

/**
 * Setup Active Navigation Highlight on Scroll
 */
function setupScrollSpy() {
  const sections = document.querySelectorAll('section[id], div[id="overview"]');
  const navLinks = document.querySelectorAll('.nav');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.scrollY + 120;

    sections.forEach((section) => {
      if (section.offsetTop <= scrollPos) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/**
 * Initialize Application on DOM Ready
 */
document.addEventListener('DOMContentLoaded', () => {
  setupDragAndDrop();
  setupScrollSpy();
  onAlgorithmChange();

  // Initial analysis on page load
  analyze();
});
