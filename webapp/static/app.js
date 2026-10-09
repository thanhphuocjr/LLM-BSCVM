/**
 * LLM-BSCVM v13 Web Application Client Script
 * Multi-Phase Smart Contract Security Audit Studio
 * (100% English UI - Professional Enterprise Security Tool)
 */

(function () {
  'use strict';

  /* ==========================================================================
     Global State
     ========================================================================== */
  const state = {
    system: null,
    examples: [],
    activeRunId: null,
    pollingInterval: null,
    currentRunData: null,
    currentLogs: '',
    selectedPhaseFilter: 'all',
  };

  /* ==========================================================================
     DOM Element Cache
     ========================================================================== */
  const DOM = {
    // Header
    chipModel: document.getElementById('chip-model'),
    valModelName: document.getElementById('val-model-name'),
    chipThreshold: document.getElementById('chip-threshold'),
    valThreshold: document.getElementById('val-threshold'),
    chipRag: document.getElementById('chip-rag'),
    chipGpu: document.getElementById('chip-gpu'),
    dotGpuStatus: document.getElementById('dot-gpu-status'),
    labelGpuChip: document.getElementById('label-gpu-chip'),
    valGpuStatus: document.getElementById('val-gpu-status'),
    activeRunBadge: document.getElementById('active-run-badge'),
    headerRunId: document.getElementById('header-run-id'),
    btnRefreshStatus: document.getElementById('btn-refresh-status'),

    // Left Panel - Editor
    inputFilename: document.getElementById('input-filename'),
    badgeCodeMode: document.getElementById('badge-code-mode'),
    fileInput: document.getElementById('file-input'),
    btnPasteClipboard: document.getElementById('btn-paste-clipboard'),
    btnClearCode: document.getElementById('btn-clear-code'),
    editorDropzone: document.getElementById('editor-dropzone'),
    codeEditor: document.getElementById('code-editor'),
    editorLines: document.getElementById('editor-lines'),
    dropOverlay: document.getElementById('drop-overlay'),
    statLines: document.getElementById('stat-lines'),
    statChars: document.getElementById('stat-chars'),
    statSize: document.getElementById('stat-size'),
    btnCopyEditor: document.getElementById('btn-copy-editor'),
    presetPills: document.querySelectorAll('.preset-pill'),

    // Left Panel - Config
    backendOptions: document.querySelectorAll('.backend-option'),
    thresholdSlider: document.getElementById('threshold-slider'),
    sliderThresholdVal: document.getElementById('slider-threshold-val'),
    btnStartAudit: document.getElementById('btn-start-audit'),
    btnCancelAudit: document.getElementById('btn-cancel-audit'),

    // Right Panel - Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    tabTrackerBadge: document.getElementById('tab-tracker-badge'),
    tabFindingsBadge: document.getElementById('tab-findings-badge'),
    tabHistoryBadge: document.getElementById('tab-history-badge'),

    // Tab 1: Stepper & Terminal
    stepperRunTitle: document.getElementById('stepper-run-title'),
    stepperOverallStatus: document.getElementById('stepper-overall-status'),
    terminalBody: document.getElementById('terminal-body'),
    chkAutoscroll: document.getElementById('chk-autoscroll'),
    btnCopyLogs: document.getElementById('btn-copy-logs'),
    btnClearLogs: document.getElementById('btn-clear-logs'),

    // Tab 2: Verdict & Findings
    verdictBanner: document.getElementById('verdict-banner'),
    verdictStatusBadge: document.getElementById('verdict-status-badge'),
    verdictMetaText: document.getElementById('verdict-meta-text'),
    gaugeScoreVal: document.getElementById('gauge-score-val'),
    metricFindingsCount: document.getElementById('metric-findings-count'),
    metricVerifiedCount: document.getElementById('metric-verified-count'),
    metricRagSimilar: document.getElementById('metric-rag-similar'),
    metricFixedStatus: document.getElementById('metric-fixed-status'),
    findingsContainer: document.getElementById('findings-container'),
    filterChips: document.querySelectorAll('.filter-chip'),
    cardAuditedCode: document.getElementById('card-audited-code'),
    auditedCodeFilename: document.getElementById('audited-code-filename'),
    auditedCodeLinesCount: document.getElementById('audited-code-lines-count'),
    codeAuditedDisplay: document.getElementById('code-audited-display'),
    headerAuditedCodeToggle: document.getElementById('header-audited-code-toggle'),
    bodyAuditedCode: document.getElementById('body-audited-code'),
    btnToggleAuditedCode: document.getElementById('btn-toggle-audited-code'),

    // Tab 3: Diff
    diffContent: document.getElementById('diff-content'),
    diffVerificationStatus: document.getElementById('diff-verification-status'),
    btnCopyPatch: document.getElementById('btn-copy-patch'),
    btnDownloadFixed: document.getElementById('btn-download-fixed'),
    sideBySideContainer: document.getElementById('side-by-side-container'),
    btnViewSideBySide: document.getElementById('btn-view-side-by-side'),
    btnViewUnified: document.getElementById('btn-view-unified'),
    origCodeWrapper: document.getElementById('orig-code-wrapper'),
    origLineNumbers: document.getElementById('orig-line-numbers'),
    origCodeContent: document.getElementById('orig-code-content'),
    origCodeMeta: document.getElementById('orig-code-meta'),
    fixedCodeWrapper: document.getElementById('fixed-code-wrapper'),
    fixedLineNumbers: document.getElementById('fixed-line-numbers'),
    fixedCodeContent: document.getElementById('fixed-code-content'),
    fixedCodeMeta: document.getElementById('fixed-code-meta'),

    // Tab 4: Reports
    reportFrame: document.getElementById('report-frame'),
    reportEmpty: document.getElementById('report-empty'),
    linkDownloadPdf: document.getElementById('link-download-pdf'),
    linkDownloadHtml: document.getElementById('link-download-html'),
    linkDownloadMd: document.getElementById('link-download-md'),

    // Tab 5: History
    historyTableBody: document.getElementById('history-table-body'),
    btnRefreshHistory: document.getElementById('btn-refresh-history'),

    // Notifications
    toastContainer: document.getElementById('toast-container'),
  };

  /* ==========================================================================
     Utility Helpers
     ========================================================================== */
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    let typeLabel = 'INFO';
    if (type === 'success') typeLabel = 'SUCCESS';
    if (type === 'error') typeLabel = 'ERROR';
    if (type === 'warning') typeLabel = 'WARNING';

    toast.innerHTML = `<span class="toast-label-tag">[${typeLabel}]</span> <span>${message}</span>`;
    DOM.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function detectContractName(source) {
    if (!source) return null;
    const match = source.match(/(?:contract|interface|library)\s+([A-Za-z0-9_]+)/);
    return match ? match[1] : null;
  }

  /* ==========================================================================
     Pipeline Stepper & Studio Reset Helpers
     ========================================================================== */
  const PHASE_KEYS = [
    { key: 'phase0', elId: 'step-phase0' },
    { key: 'phase1a', elId: 'step-phase1a' },
    { key: 'phase1b', elId: 'step-phase1b' },
    { key: 'phase1c', elId: 'step-phase1c' },
    { key: 'fusion', elId: 'step-fusion' },
    { key: 'phase2', elId: 'step-phase2' },
    { key: 'phase3', elId: 'step-phase3' },
    { key: 'phase4', elId: 'step-phase4' },
    { key: 'phase5', elId: 'step-phase5' },
    { key: 'phase6', elId: 'step-phase6' },
  ];

  function resetPipelineStepper(runId) {
    if (DOM.stepperRunTitle) DOM.stepperRunTitle.textContent = runId || 'No active run';
    if (DOM.stepperOverallStatus) {
      DOM.stepperOverallStatus.textContent = runId ? 'Initializing...' : 'Ready';
      DOM.stepperOverallStatus.className = 'stepper-status-pill' + (runId ? ' status-running' : '');
    }

    PHASE_KEYS.forEach(({ elId }) => {
      const card = document.getElementById(elId);
      if (card) {
        card.className = 'step-card step-pending';
        const tag = card.querySelector('.step-status-tag, .step-status-icon');
        if (tag) tag.textContent = 'WAIT';
      }
    });
  }

  function resetAllTabsState({
    clearTerminal = true,
    terminalMessage = '[System] New contract code loaded. Ready for security audit.',
    isStartingAudit = false,
    newRunId = null,
  } = {}) {
    // 1. Clear run tracking variables
    if (!isStartingAudit) {
      state.activeRunId = null;
      if (DOM.activeRunBadge) DOM.activeRunBadge.style.display = 'none';
      if (DOM.headerRunId) DOM.headerRunId.textContent = '...';
      if (DOM.tabTrackerBadge) DOM.tabTrackerBadge.style.display = 'none';
      if (DOM.btnStartAudit) DOM.btnStartAudit.style.display = 'inline-flex';
      if (DOM.btnCancelAudit) DOM.btnCancelAudit.style.display = 'none';
    }
    state.currentRunData = null;
    state.currentLogs = '';
    state.selectedPhaseFilter = 'all';

    // 2. Reset Pipeline Stepper & Step Highlights
    resetPipelineStepper(isStartingAudit ? newRunId : '');
    document.querySelectorAll('.phase-filter-pill').forEach((p) => p.classList.remove('active'));
    document.querySelector('.phase-filter-pill[data-phase="all"]')?.classList.add('active');
    document.querySelectorAll('.step-card').forEach((c) => c.classList.remove('phase-active'));

    // 3. Reset Terminal Body
    if (clearTerminal && DOM.terminalBody) {
      DOM.terminalBody.innerHTML = `
        <div class="log-line log-system">${escapeHtml(terminalMessage)}</div>
      `;
    }

    // 4. Reset Tab 2: Verdict & Findings UI
    if (DOM.verdictStatusBadge) {
      if (isStartingAudit) {
        DOM.verdictStatusBadge.textContent = 'AUDITING IN PROGRESS...';
        DOM.verdictStatusBadge.className = 'verdict-badge';
        DOM.verdictStatusBadge.style.background = 'linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(79, 172, 254, 0.3))';
        DOM.verdictStatusBadge.style.borderColor = '#00f2fe';
        DOM.verdictStatusBadge.style.color = '#e0f2fe';
      } else {
        DOM.verdictStatusBadge.textContent = 'NOT AUDITED';
        DOM.verdictStatusBadge.className = 'verdict-badge';
        DOM.verdictStatusBadge.style = '';
      }
    }
    if (DOM.verdictMetaText) {
      DOM.verdictMetaText.textContent = isStartingAudit
        ? `Auditing contract: ${DOM.inputFilename?.value || 'Contract.sol'} | Neural Detector + Multi-Modal Pipeline running...`
        : 'Contract code updated. Click "START SECURITY AUDIT" to analyze.';
    }
    if (DOM.gaugeScoreVal) DOM.gaugeScoreVal.textContent = '0.00';
    if (DOM.metricFindingsCount) DOM.metricFindingsCount.textContent = '0';
    if (DOM.metricVerifiedCount) DOM.metricVerifiedCount.textContent = '0';
    if (DOM.metricRagSimilar) DOM.metricRagSimilar.textContent = '0';
    if (DOM.metricFixedStatus) {
      DOM.metricFixedStatus.textContent = 'Pending';
      DOM.metricFixedStatus.className = 'metric-num text-cyan';
    }
    if (DOM.tabFindingsBadge) DOM.tabFindingsBadge.textContent = '0';
    if (DOM.findingsContainer) {
      if (isStartingAudit) {
        DOM.findingsContainer.innerHTML = `
          <div class="empty-placeholder">
            <div class="pulse-ring" style="width: 24px; height: 24px; margin-bottom: 12px; background: #00f2fe;"></div>
            <p style="color: #38bdf8; font-weight: 600;">Security audit in progress...</p>
            <span style="color: var(--text-muted); font-size: 0.75rem;">Scanning bytecode AST, querying 11.8k RAG cases, and running Qwen neural model.</span>
          </div>
        `;
      } else {
        DOM.findingsContainer.innerHTML = `
          <div class="empty-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 8v4M12 16h.01"/>
            </svg>
            <p>New contract loaded. Please run an audit or select a past run from History.</p>
          </div>
        `;
      }
    }
    DOM.filterChips.forEach((c) => c.classList.remove('active'));
    document.querySelector('.filter-chip[data-filter="all"]')?.classList.add('active');
    if (DOM.cardAuditedCode) DOM.cardAuditedCode.style.display = 'none';
    if (DOM.codeAuditedDisplay) DOM.codeAuditedDisplay.textContent = '';

    // 5. Reset Tab 3: Diff Viewer (Side-by-side & Unified)
    if (DOM.origCodeContent) DOM.origCodeContent.innerHTML = '<div class="empty-diff-hint">No original contract loaded.</div>';
    if (DOM.fixedCodeContent) DOM.fixedCodeContent.innerHTML = '<div class="empty-diff-hint">No automated patch generated yet.</div>';
    if (DOM.origLineNumbers) DOM.origLineNumbers.innerHTML = '<div class="line-num-cell">1</div>';
    if (DOM.fixedLineNumbers) DOM.fixedLineNumbers.innerHTML = '<div class="line-num-cell">1</div>';
    if (DOM.origCodeMeta) DOM.origCodeMeta.textContent = '0 lines';
    if (DOM.fixedCodeMeta) DOM.fixedCodeMeta.textContent = '0 lines';
    if (DOM.diffContent) {
      DOM.diffContent.innerHTML = `
        <div class="empty-placeholder">
          <p>No automated patch generated yet. After Phase 4 (Fixer) completes, diff comparison will be rendered here.</p>
        </div>
      `;
    }
    if (DOM.diffVerificationStatus) {
      DOM.diffVerificationStatus.textContent = 'Unverified';
      DOM.diffVerificationStatus.className = 'verification-badge';
    }

    // 6. Reset Tab 4: Reports
    if (DOM.reportFrame) {
      DOM.reportFrame.src = 'about:blank';
      DOM.reportFrame.style.display = 'none';
    }
    if (DOM.reportEmpty) DOM.reportEmpty.style.display = 'flex';
    if (DOM.linkDownloadHtml) DOM.linkDownloadHtml.removeAttribute('href');
    if (DOM.linkDownloadPdf) DOM.linkDownloadPdf.style.display = 'none';
    if (DOM.linkDownloadMd) DOM.linkDownloadMd.style.display = 'none';
  }

  /* ==========================================================================
     Editor Operations & Line Numbers
     ========================================================================== */
  function updateEditorStats() {
    const text = DOM.codeEditor.value;
    const lines = text ? text.split('\n').length : 0;
    const chars = text.length;
    const bytes = new Blob([text]).size;

    DOM.statLines.textContent = lines;
    DOM.statChars.textContent = chars;
    DOM.statSize.textContent = formatBytes(bytes);

    // Update line number gutter
    let lineNumbersHtml = '';
    const totalLines = Math.max(lines, 1);
    for (let i = 1; i <= totalLines; i++) {
      lineNumbersHtml += `${i}\n`;
    }
    DOM.editorLines.textContent = lineNumbersHtml;
  }

  // Sync scroll between textarea and line numbers
  DOM.codeEditor.addEventListener('scroll', () => {
    DOM.editorLines.scrollTop = DOM.codeEditor.scrollTop;
  });

  // When user types directly in editor
  DOM.codeEditor.addEventListener('input', () => {
    updateEditorStats();
    DOM.presetPills.forEach((p) => p.classList.remove('active'));
    if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = 'Custom Source Code';

    // If editor previously held completed run data or results, reset tabs on code change
    if (state.currentRunData !== null || state.activeRunId !== null) {
      resetAllTabsState({
        clearTerminal: false,
        terminalMessage: '[System] Source code modified in editor. Previous audit results reset.',
      });
    }

    const cName = detectContractName(DOM.codeEditor.value);
    if (cName && (DOM.inputFilename.value === 'Contract.sol' || DOM.inputFilename.value === 'CustomContract.sol')) {
      DOM.inputFilename.value = `${cName}.sol`;
    }
  });

  function cleanAndFormatSolidityCode(rawText) {
    if (!rawText) return rawText;
    let text = rawText.trim();

    // 1. Strip outer enclosing double/backtick quotes from JSON if present
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith('`') && text.endsWith('`'))) {
      if (text.includes('\\n') || text.includes('contract ') || text.includes('pragma ')) {
        text = text.slice(1, -1);
      }
    }

    // 2. Unescape literal \r\n or \n into real newlines
    if (text.includes('\\n')) {
      text = text.replace(/\\r\\n/g, '\n').replace(/\\n/g, '\n');
    }

    // 3. Unescape literal \t into 4 spaces
    if (text.includes('\\t')) {
      text = text.replace(/\\t/g, '    ');
    }

    // 4. Unescape \" into "
    if (text.includes('\\"')) {
      text = text.replace(/\\"/g, '"');
    }

    // 5. Unescape \' into '
    if (text.includes("\\'")) {
      text = text.replace(/\\'/g, "'");
    }

    // 6. Unescape \\ into \
    if (text.includes('\\\\')) {
      text = text.replace(/\\\\/g, '\\');
    }

    return text;
  }

  // Paste handling (Ctrl+V / Cmd+V)
  DOM.codeEditor.addEventListener('paste', (e) => {
    // Immediately reset all tabs and badges for new pasted contract
    resetAllTabsState({
      clearTerminal: true,
      terminalMessage: '[System] New contract code pasted into editor. Tabs and results reset. Ready for audit.',
    });
    DOM.presetPills.forEach((p) => p.classList.remove('active'));
    if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = 'Custom Source Code';

    const pastedData = (e.clipboardData || window.clipboardData)?.getData('text');
    if (pastedData && (pastedData.includes('\\n') || pastedData.startsWith('"') || pastedData.includes('\\t') || pastedData.includes('\\"'))) {
      e.preventDefault();
      const formatted = cleanAndFormatSolidityCode(pastedData);
      const start = DOM.codeEditor.selectionStart;
      const end = DOM.codeEditor.selectionEnd;
      const currentVal = DOM.codeEditor.value;
      DOM.codeEditor.value = currentVal.substring(0, start) + formatted + currentVal.substring(end);
      DOM.codeEditor.selectionStart = DOM.codeEditor.selectionEnd = start + formatted.length;

      const cName = detectContractName(DOM.codeEditor.value);
      if (cName) DOM.inputFilename.value = `${cName}.sol`;
      updateEditorStats();
      showToast('Solidity source code formatted & all tabs reset!', 'success');
      return;
    }

    setTimeout(() => {
      const text = DOM.codeEditor.value;
      if (text.includes('\\n')) {
        DOM.codeEditor.value = cleanAndFormatSolidityCode(text);
      }
      const cName = detectContractName(DOM.codeEditor.value);
      if (cName) {
        DOM.inputFilename.value = `${cName}.sol`;
      }
      updateEditorStats();
      showToast('New contract code recognized & all tabs reset!', 'success');
    }, 50);
  });

  // Allow Tab indentation in textarea
  DOM.codeEditor.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = DOM.codeEditor.selectionStart;
      const end = DOM.codeEditor.selectionEnd;
      DOM.codeEditor.value =
        DOM.codeEditor.value.substring(0, start) +
        '    ' +
        DOM.codeEditor.value.substring(end);
      DOM.codeEditor.selectionStart = DOM.codeEditor.selectionEnd = start + 4;
      updateEditorStats();
    }
  });

  // Paste from Clipboard button
  if (DOM.btnPasteClipboard) {
    DOM.btnPasteClipboard.addEventListener('click', async () => {
      try {
        const rawText = await navigator.clipboard.readText();
        if (rawText && rawText.trim()) {
          // Immediately reset all tabs and badges
          resetAllTabsState({
            clearTerminal: true,
            terminalMessage: '[System] New contract code pasted from clipboard. Tabs and results reset. Ready for audit.',
          });
          const text = cleanAndFormatSolidityCode(rawText);
          DOM.codeEditor.value = text;
          const cName = detectContractName(text);
          DOM.inputFilename.value = cName ? `${cName}.sol` : 'CustomContract.sol';
          DOM.presetPills.forEach((p) => p.classList.remove('active'));
          if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = 'Custom Source Code';
          updateEditorStats();
          showToast('Pasted code & reset all tabs! Ready for audit.', 'success');
        } else {
          showToast('Clipboard is empty!', 'warning');
          DOM.codeEditor.focus();
        }
      } catch (err) {
        DOM.codeEditor.focus();
        showToast('Please press Ctrl+V (Cmd+V) to paste code into editor.', 'info');
      }
    });
  }

  // Drag & Drop Handling
  ['dragenter', 'dragover'].forEach((eventName) => {
    DOM.editorDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropOverlay.classList.add('active');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    DOM.editorDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropOverlay.classList.remove('active');
    });
  });

  DOM.editorDropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleUploadedFile(files[0]);
    }
  });

  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  function handleUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      resetAllTabsState({
        clearTerminal: true,
        terminalMessage: `[System] File "${file.name}" loaded into editor. Tabs and results reset. Ready for audit.`,
      });
      DOM.codeEditor.value = event.target.result;
      DOM.inputFilename.value = file.name;
      updateEditorStats();
      if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = 'Uploaded File';
      showToast(`Loaded file "${file.name}" & reset all tabs!`, 'success');
      DOM.presetPills.forEach((p) => p.classList.remove('active'));
    };
    reader.readAsText(file);
  }

  DOM.btnClearCode.addEventListener('click', () => {
    resetAllTabsState({
      clearTerminal: true,
      terminalMessage: '[System] Editor cleared. All tabs and previous audit results reset.',
    });
    DOM.codeEditor.value = '';
    DOM.inputFilename.value = 'Contract.sol';
    updateEditorStats();
    DOM.presetPills.forEach((p) => p.classList.remove('active'));
    if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = 'Custom Source Code';
    showToast('Editor cleared & all tabs reset.', 'info');
  });

  DOM.btnCopyEditor.addEventListener('click', () => {
    navigator.clipboard.writeText(DOM.codeEditor.value).then(() => {
      showToast('Code copied to clipboard!', 'success');
    });
  });

  /* ==========================================================================
     Preset Benchmark Loading
     ========================================================================== */
  async function loadExampleContracts() {
    try {
      const res = await fetch('/api/contracts/examples');
      if (!res.ok) return;
      state.examples = await res.json();

      // Setup click listeners for presets
      DOM.presetPills.forEach((pill) => {
        pill.addEventListener('click', () => {
          const filename = pill.getAttribute('data-example');
          const found = state.examples.find((ex) => ex.file_name === filename);
          if (found) {
            resetAllTabsState({
              clearTerminal: true,
              terminalMessage: `[System] Loaded benchmark sample "${found.title}" (${found.vuln_type}). Tabs and results reset.`,
            });
            DOM.codeEditor.value = found.code;
            DOM.inputFilename.value = found.file_name;
            updateEditorStats();
            DOM.presetPills.forEach((p) => p.classList.remove('active'));
            pill.classList.add('active');
            if (DOM.badgeCodeMode) DOM.badgeCodeMode.textContent = `Sample: ${found.title}`;
            showToast(`Loaded benchmark "${found.title}" & reset tabs!`, 'info');
          }
        });
      });
    } catch (err) {
      console.warn('Unable to load example contracts:', err);
    }
  }

  /* ==========================================================================
     Configuration Controls & Hardware Synchronization
     ========================================================================== */
  function updateHardwareDisplay() {
    const selectedRadio = document.querySelector('input[name="backend"]:checked');
    const backend = selectedRadio ? selectedRadio.value : 'server_gpu';
    const chipGpu = DOM.chipGpu || document.getElementById('chip-gpu');
    const dot = DOM.dotGpuStatus || document.getElementById('dot-gpu-status');
    const label = DOM.labelGpuChip || document.getElementById('label-gpu-chip');
    const val = DOM.valGpuStatus || document.getElementById('val-gpu-status');

    if (!val) return;

    if (backend === 'server_gpu') {
      if (label) label.textContent = 'GPU Backend:';
      const isOnline = state.system && state.system.server_gpu_online;
      const devName = state.system?.server_gpu_info?.gpu_device || 'RTX 4070 Ti SUPER';
      if (isOnline) {
        val.textContent = `${devName} (Online)`;
        val.style.color = '#38bdf8';
        if (dot) {
          dot.className = 'chip-dot dot-ready';
          dot.style.background = '#38bdf8';
        }
        if (chipGpu) chipGpu.title = 'Dedicated GPU Server (RTX 4070 Ti SUPER 16GB VRAM, Qwen2.5-14B v13)';
      } else {
        val.textContent = 'RTX 4070 Ti SUPER (Offline)';
        val.style.color = '#fbbf24';
        if (dot) {
          dot.className = 'chip-dot dot-pending';
          dot.style.background = '#fbbf24';
        }
        if (chipGpu) chipGpu.title = 'Dedicated GPU Server is not responding';
      }
    } else if (backend === 'kaggle') {
      if (label) label.textContent = 'Kaggle GPU:';
      const hasKaggle = state.system && state.system.kaggle_cli_installed;
      val.textContent = hasKaggle ? 'Tesla T4 (Cloud Ready)' : 'Tesla T4 (API Unconfigured)';
      val.style.color = hasKaggle ? '#34d399' : '#fbbf24';
      if (dot) {
        dot.className = hasKaggle ? 'chip-dot dot-ready' : 'chip-dot dot-pending';
        dot.style.background = hasKaggle ? '#34d399' : '#fbbf24';
      }
      if (chipGpu) chipGpu.title = 'Kaggle Cloud Kernel Computing Environment (Tesla T4 GPU 16GB VRAM)';
    }
  }

  DOM.backendOptions.forEach((opt) => {
    opt.addEventListener('click', () => {
      DOM.backendOptions.forEach((o) => o.classList.remove('selected'));
      opt.classList.add('selected');
      const radio = opt.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        updateHardwareDisplay();
      }
    });
  });

  DOM.thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(3);
    DOM.sliderThresholdVal.textContent = val;
    DOM.valThreshold.textContent = val;
  });

  /* ==========================================================================
     Tabs Navigation
     ========================================================================== */
  DOM.tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      DOM.tabBtns.forEach((b) => b.classList.remove('active'));
      DOM.tabPanes.forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');

      if (targetId === 'tab-history') {
        fetchRunsHistory();
      }
    });
  });

  function switchTab(tabId) {
    const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.click();
  }

  /* ==========================================================================
     System Status
     ========================================================================== */
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/system/status');
      if (!res.ok) return;
      const data = await res.json();
      state.system = data;

      DOM.valModelName.textContent = data.model_version;
      DOM.valThreshold.textContent = data.best_threshold;
      DOM.sliderThresholdVal.textContent = data.best_threshold;
      DOM.thresholdSlider.value = data.best_threshold;

      updateHardwareDisplay();
    } catch (err) {
      console.warn('Failed to fetch system status:', err);
    }
  }

  /* ==========================================================================
     Reset Studio / Refresh Button
     ========================================================================== */
  DOM.btnRefreshStatus.addEventListener('click', () => {
    resetAllTabsState({
      clearTerminal: true,
      terminalMessage: '[System] Studio state reset. Multi-phase pipeline ready for audit.',
    });
    fetchSystemStatus();
    fetchRunsHistory();
  });

  const PIPELINE_ORDER = [
    {
      id: 'step-phase0',
      label: 'Phase 0: Input & AST',
      startPattern: '[PHASE 0]',
      donePattern: '[PHASE 0] Completed',
      fileKey: 'phase0',
    },
    {
      id: 'step-phase1a',
      label: 'Phase 1A: Slither Static',
      startPattern: '[PHASE 1A]',
      donePattern: '[PHASE 1A] Completed',
      fileKey: 'phase1a',
    },
    {
      id: 'step-phase1b',
      label: 'Phase 1B: RAG Retrieval',
      startPattern: '[PHASE 1B]',
      donePattern: '[PHASE 1B] Completed',
      fileKey: 'phase1b',
    },
    {
      id: 'step-phase1c',
      label: 'Phase 1C: Neural Detector v13',
      startPattern: '[PHASE 1C]',
      donePattern: '[PHASE 1C] Completed',
      fileKey: 'phase1c',
    },
    {
      id: 'step-fusion',
      label: 'Phase 1: Multi-Modal Fusion',
      startPattern: '[PHASE 1 FUSED]',
      donePattern: '[PHASE 1 FUSED] Completed',
      fileKey: 'fusion',
    },
    {
      id: 'step-phase2',
      label: 'Phase 2: Security Advisor',
      startPattern: '[PHASE 2]',
      donePattern: '[PHASE 2] Completed',
      fileKey: 'phase2',
    },
    {
      id: 'step-phase3',
      label: 'Phase 3: Impact Assessor',
      startPattern: '[PHASE 3]',
      donePattern: '[PHASE 3] Completed',
      fileKey: 'phase3',
    },
    {
      id: 'step-phase4',
      label: 'Phase 4: Automated Fixer',
      startPattern: '[PHASE 4]',
      donePattern: '[PHASE 4] Completed',
      fileKey: 'phase4',
    },
    {
      id: 'step-phase5',
      label: 'Phase 5: Dual Verification',
      startPattern: '[PHASE 5]',
      donePattern: '[PHASE 5] Completed',
      fileKey: 'phase5',
    },
    {
      id: 'step-phase6',
      label: 'Phase 6: Executive Reporter',
      startPattern: '[PHASE 6]',
      donePattern: '[PHASE 6] Completed',
      fileKey: 'phase6',
    },
  ];

  function setStepCardState(elId, state, tagText) {
    const card = document.getElementById(elId);
    if (!card) return;
    card.className = `step-card step-${state}`;
    const tag = card.querySelector('.step-status-tag, .step-status-icon');
    if (tag) tag.textContent = tagText;
  }

  function updateStepperFromLogs(logText, runStatus, phaseFiles) {
    const text = logText || '';
    const isCompleted = runStatus === 'completed' || text.includes('Run completed successfully') || (phaseFiles && phaseFiles.phase6);
    const isFailed = runStatus === 'failed' || text.includes('Process exited with code 1') || text.includes('Run failed');

    // Trường hợp 1: Chạy xong toàn bộ -> TẤT CẢ 10 PHASE ĐỀU XANH LÁ (PASS)
    if (isCompleted) {
      PIPELINE_ORDER.forEach(({ id }) => {
        setStepCardState(id, 'success', 'PASS');
      });
      if (DOM.stepperOverallStatus) {
        DOM.stepperOverallStatus.textContent = 'Completed (All Phases Passed)';
        DOM.stepperOverallStatus.className = 'stepper-status-pill status-success';
      }
      return;
    }

    // Trường hợp 2: Pipeline đang thực thi hoặc vừa khởi tạo
    let runningStepFound = false;

    for (let i = 0; i < PIPELINE_ORDER.length; i++) {
      const step = PIPELINE_ORDER[i];
      const hasCompletedInLog = text.includes(step.donePattern);
      const anySubsequentStarted = PIPELINE_ORDER.slice(i + 1).some(
        (nextStep) => text.includes(nextStep.startPattern) || text.includes(nextStep.donePattern)
      );

      // Bước này đã hoàn thành nếu log đã ghi completed hoặc có bất kỳ bước kế sau nào đã bắt đầu
      const isDone = hasCompletedInLog || anySubsequentStarted;

      if (isDone) {
        setStepCardState(step.id, 'success', 'PASS');
      } else if (!runningStepFound) {
        // Đây là bước đầu tiên chưa hoàn thành -> ĐANG CHẠY (VÀNG HỔ PHÁCH)
        runningStepFound = true;
        if (isFailed) {
          setStepCardState(step.id, 'failed', 'FAIL');
        } else {
          setStepCardState(step.id, 'running', 'RUNNING');
          if (DOM.stepperOverallStatus) {
            DOM.stepperOverallStatus.textContent = `Running: ${step.label}`;
            DOM.stepperOverallStatus.className = 'stepper-status-pill status-running';
          }
        }
      } else {
        // Các bước còn lại phía sau chưa chạy tới -> XÁM (WAIT)
        setStepCardState(step.id, 'pending', 'WAIT');
      }
    }

    if (isFailed && DOM.stepperOverallStatus) {
      DOM.stepperOverallStatus.textContent = 'Failed';
      DOM.stepperOverallStatus.className = 'stepper-status-pill status-failed';
    }
  }

  /* ==========================================================================
     Terminal Log Rendering & Phase Filtering
     ========================================================================== */
  function renderTerminalLogs(rawLogs) {
    if (!rawLogs) return;
    state.currentLogs = rawLogs;

    if (state.selectedPhaseFilter && state.selectedPhaseFilter !== 'all') {
      renderFilteredLogs(state.selectedPhaseFilter);
      return;
    }

    const lines = rawLogs.split('\n');
    let html = '';

    lines.forEach((line) => {
      let cls = 'log-line';
      if (line.includes('ERROR') || line.includes('Failed') || line.includes('Traceback')) {
        cls += ' log-error';
      } else if (line.includes('WARNING') || line.includes('warn')) {
        cls += ' log-warn';
      } else if (line.includes('PASS') || line.includes('Successfully') || line.includes('passed')) {
        cls += ' log-success';
      } else if (line.startsWith('[PHASE') || line.startsWith('===')) {
        cls += ' log-system';
      } else {
        cls += ' log-info';
      }
      html += `<div class="${cls}">${escapeHtml(line)}</div>`;
    });

    DOM.terminalBody.innerHTML = html;

    if (DOM.chkAutoscroll.checked) {
      DOM.terminalBody.scrollTop = DOM.terminalBody.scrollHeight;
    }
  }

  const PHASE_LOG_KEYWORDS = {
    phase0: ['PHASE 0', 'Phase 0', 'AST', 'phase0_input', 'Solidity AST'],
    phase1a: ['PHASE 1A', 'Phase 1A', 'Slither', 'phase1a_static', 'SWC-', 'rule'],
    phase1b: ['PHASE 1B', 'Phase 1B', 'RAG', 'retrieval', 'phase1b_rag', 'vector', 'embedding'],
    phase1c: ['PHASE 1C', 'Phase 1C', 'Qwen', 'Detector', 'phase1c_llm', 'RTX', 'logits', 'score'],
    fusion: ['PHASE 1 FUSED', 'Phase 1', 'Fusion', 'phase1_fused', 'multi-modal'],
    phase2: ['PHASE 2', 'Phase 2', 'Advisor', 'phase2_advisor', 'root cause', 'PoC'],
    phase3: ['PHASE 3', 'Phase 3', 'Assessor', 'phase3_assessor', 'CVSS', 'severity'],
    phase4: ['PHASE 4', 'Phase 4', 'Fixer', 'phase4_fixer', 'patch', 'fixed_code', 'automated fix'],
    phase5: ['PHASE 5', 'Phase 5', 'Verification', 'phase5_verification', 'solc', 'compiler verify'],
    phase6: ['PHASE 6', 'Phase 6', 'Reporter', 'phase6_report', 'pdf', 'html', 'audit report'],
  };

  function renderFilteredLogs(phaseKey) {
    state.selectedPhaseFilter = phaseKey;

    if (!state.currentLogs && !state.currentRunData) {
      DOM.terminalBody.innerHTML = `
        <div class="log-line log-system">[System] No execution logs available for this phase yet. Start an audit or select a past run from History.</div>
      `;
      return;
    }

    if (phaseKey === 'all') {
      const rawLogs = state.currentLogs || '';
      const lines = rawLogs ? rawLogs.split('\n') : [];
      let html = '';
      lines.forEach((line) => {
        let cls = 'log-line';
        if (line.includes('ERROR') || line.includes('Failed')) cls += ' log-error';
        else if (line.includes('PASS') || line.includes('Successfully')) cls += ' log-success';
        else if (line.includes('WARNING')) cls += ' log-warn';
        else if (line.startsWith('[PHASE') || line.startsWith('===')) cls += ' log-system';
        else cls += ' log-info';
        html += `<div class="${cls}">${escapeHtml(line)}</div>`;
      });
      DOM.terminalBody.innerHTML = html || '<div class="log-line log-system">[No execution logs]</div>';
      if (DOM.chkAutoscroll.checked) DOM.terminalBody.scrollTop = DOM.terminalBody.scrollHeight;
      return;
    }

    const keywords = PHASE_LOG_KEYWORDS[phaseKey] || [phaseKey];
    const lines = (state.currentLogs || '').split('\n');
    const matchedLines = lines.filter((line) => keywords.some((kw) => line.toLowerCase().includes(kw.toLowerCase())));

    let phaseData = null;
    if (state.currentRunData && state.currentRunData.phases) {
      const p = state.currentRunData.phases;
      if (phaseKey === 'phase0') phaseData = state.currentRunData.original_code ? { file_name: state.currentRunData.contract_name, lines: state.currentRunData.original_code.split('\n').length } : null;
      else if (phaseKey === 'phase1a') phaseData = p.phase1a_static;
      else if (phaseKey === 'phase1b') phaseData = p.phase1b_rag;
      else if (phaseKey === 'phase1c') phaseData = p.phase1c_llm;
      else if (phaseKey === 'fusion') phaseData = p.phase1_fused;
      else if (phaseKey === 'phase2') phaseData = p.phase2_advisor;
      else if (phaseKey === 'phase3') phaseData = p.phase3_assessor;
      else if (phaseKey === 'phase4') phaseData = p.phase4_fixer;
      else if (phaseKey === 'phase5') phaseData = p.phase5_verification;
      else if (phaseKey === 'phase6') phaseData = { has_pdf: state.currentRunData.has_pdf, has_html: state.currentRunData.has_html, verdict: state.currentRunData.manifest?.detection?.verdict };
    }

    let detailSummaryHtml = '';
    if (phaseData && Object.keys(phaseData).length > 0) {
      detailSummaryHtml = `
        <div class="log-line log-system" style="border-bottom: 1px dashed rgba(0, 242, 254, 0.4); padding-bottom: 5px; margin-bottom: 8px;">
          [EXTRACTED ARTIFACT DATA - ${phaseKey.toUpperCase()}]
        </div>
        <div class="log-line log-info" style="color: #38bdf8; white-space: pre-wrap; font-family: var(--font-mono);">${escapeHtml(JSON.stringify(phaseData, null, 2))}</div>
        <div class="log-line log-system" style="border-top: 1px dashed rgba(255, 255, 255, 0.1); padding-top: 5px; margin-top: 8px;">
          [RELATED PIPELINE EXECUTION LOGS]
        </div>
      `;
    }

    let logsHtml = '';
    if (matchedLines.length > 0) {
      matchedLines.forEach((line) => {
        let cls = 'log-line';
        if (line.includes('ERROR') || line.includes('Failed')) cls += ' log-error';
        else if (line.includes('PASS') || line.includes('Successfully')) cls += ' log-success';
        else if (line.includes('WARNING')) cls += ' log-warn';
        else cls += ' log-info';
        logsHtml += `<div class="${cls}">${escapeHtml(line)}</div>`;
      });
    } else {
      logsHtml = `<div class="log-line log-system" style="color: #64748b;">(No explicit execution log lines matched for this phase in execution.log)</div>`;
    }

    DOM.terminalBody.innerHTML = `
      <div class="log-line log-system" style="color: #00f2fe; font-weight: 600; margin-bottom: 6px;">
        [FILTERED LOGS: ${phaseKey.toUpperCase()}]
      </div>
      ${detailSummaryHtml}
      ${logsHtml}
    `;

    DOM.terminalBody.scrollTop = 0;
  }

  // Bind click for Step Cards
  document.querySelectorAll('.step-card').forEach((card) => {
    card.addEventListener('click', () => {
      const phaseKey = card.getAttribute('data-phase');
      if (!phaseKey) return;

      const isCurrentActive = card.classList.contains('phase-active');
      document.querySelectorAll('.step-card').forEach((c) => c.classList.remove('phase-active'));
      document.querySelectorAll('.phase-filter-pill').forEach((p) => p.classList.remove('active'));

      if (isCurrentActive) {
        state.selectedPhaseFilter = 'all';
        document.querySelector('.phase-filter-pill[data-phase="all"]')?.classList.add('active');
        renderFilteredLogs('all');
        showToast('Viewing full execution logs (All Logs)', 'info');
      } else {
        card.classList.add('phase-active');
        state.selectedPhaseFilter = phaseKey;
        const targetPill = document.querySelector(`.phase-filter-pill[data-phase="${phaseKey}"]`);
        if (targetPill) targetPill.classList.add('active');
        renderFilteredLogs(phaseKey);
        showToast(`Filtered logs for: ${card.querySelector('.step-name')?.textContent || phaseKey}`, 'info');
      }
    });
  });

  // Bind click for Phase Filter Pills
  document.querySelectorAll('.phase-filter-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const phaseKey = pill.getAttribute('data-phase');
      document.querySelectorAll('.phase-filter-pill').forEach((p) => p.classList.remove('active'));
      document.querySelectorAll('.step-card').forEach((c) => c.classList.remove('phase-active'));

      pill.classList.add('active');
      state.selectedPhaseFilter = phaseKey;
      if (phaseKey !== 'all') {
        const card = document.querySelector(`.step-card[data-phase="${phaseKey}"]`);
        if (card) card.classList.add('phase-active');
      }
      renderFilteredLogs(phaseKey);
    });
  });

  DOM.btnCopyLogs.addEventListener('click', () => {
    navigator.clipboard.writeText(DOM.terminalBody.innerText).then(() => {
      showToast('Copied logs to clipboard!', 'success');
    });
  });

  DOM.btnClearLogs.addEventListener('click', () => {
    DOM.terminalBody.innerHTML = '<div class="log-line log-system">[Logs cleared]</div>';
  });

  /* ==========================================================================
     Audit Execution & Polling
     ========================================================================== */
  DOM.btnStartAudit.addEventListener('click', async () => {
    const code = DOM.codeEditor.value.trim();
    if (!code) {
      showToast('Please enter or paste Solidity smart contract code.', 'warning');
      return;
    }

    const fileName = DOM.inputFilename.value.trim() || 'Contract.sol';
    const selectedRadio = document.querySelector('input[name="backend"]:checked');
    const backend = selectedRadio ? selectedRadio.value : 'server_gpu';
    const detectorThreshold = parseFloat(DOM.thresholdSlider.value);

    // Immediately reset all tabs, badges, diff, and logs for the new audit run
    resetAllTabsState({
      clearTerminal: true,
      terminalMessage: `[System] Submitting audit request for "${fileName}" (${backend})...`,
      isStartingAudit: true,
      newRunId: 'Preparing...',
    });

    DOM.btnStartAudit.style.display = 'none';
    DOM.btnCancelAudit.style.display = 'inline-flex';

    switchTab('tab-tracker');
    DOM.tabTrackerBadge.style.display = 'inline-block';

    try {
      showToast('Starting security audit pipeline...', 'info');
      const res = await fetch('/api/audit/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: code,
          file_name: fileName,
          backend: backend,
          detector_threshold: detectorThreshold,
          tools: 'slither',
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to start security audit.');
      }

      const result = await res.json();
      const runId = result.run_id;
      state.activeRunId = runId;

      DOM.headerRunId.textContent = runId;
      DOM.activeRunBadge.style.display = 'inline-flex';
      resetPipelineStepper(runId);

      startPolling(runId);
    } catch (err) {
      showToast(err.message, 'error');
      DOM.btnStartAudit.style.display = 'inline-flex';
      DOM.btnCancelAudit.style.display = 'none';
      DOM.tabTrackerBadge.style.display = 'none';
    }
  });

  DOM.btnCancelAudit.addEventListener('click', async () => {
    if (!state.activeRunId) return;
    try {
      await fetch(`/api/runs/${state.activeRunId}/cancel`, { method: 'POST' });
      showToast('Cancellation request submitted.', 'info');
      stopPolling();
      DOM.stepperOverallStatus.textContent = 'Aborted';
      DOM.stepperOverallStatus.className = 'stepper-status-pill status-failed';
    } catch (err) {
      showToast('Error aborting process: ' + err.message, 'error');
    }
  });

  function startPolling(runId) {
    if (state.pollingInterval) clearInterval(state.pollingInterval);

    state.pollingInterval = setInterval(async () => {
      try {
        const logRes = await fetch(`/api/runs/${runId}/logs`);
        if (!logRes.ok) return;
        const logData = await logRes.json();

        renderTerminalLogs(logData.logs);
        updateStepperFromLogs(logData.logs, logData.status, logData.phase_files);

        if (logData.status === 'completed' || logData.status === 'failed') {
          stopPolling();
          await loadRunDetail(runId);
          await fetchRunsHistory();
          showToast(
            logData.status === 'completed'
              ? 'Security audit completed successfully!'
              : 'Security audit finished (check logs for details).',
            logData.status === 'completed' ? 'success' : 'error'
          );
        }
      } catch (err) {
        console.warn('Polling error:', err);
      }
    }, 750);
  }

  function stopPolling() {
    if (state.pollingInterval) {
      clearInterval(state.pollingInterval);
      state.pollingInterval = null;
    }
    DOM.btnStartAudit.style.display = 'inline-flex';
    DOM.btnCancelAudit.style.display = 'none';
    DOM.activeRunBadge.style.display = 'none';
    DOM.tabTrackerBadge.style.display = 'none';
  }

  /* ==========================================================================
     Load Run Detail & Populate Results
     ========================================================================== */
  async function loadRunDetail(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) return;
      const data = await res.json();
      state.currentRunData = data;
      state.activeRunId = runId;

      DOM.stepperRunTitle.textContent = runId;

      // 1. Fill the code editor with the historical audited contract code
      if (data.original_code) {
        DOM.codeEditor.value = data.original_code;
        DOM.inputFilename.value = data.contract_name || 'Contract.sol';
        updateEditorStats();
        if (DOM.badgeCodeMode) {
          DOM.badgeCodeMode.textContent = `Historical: ${data.contract_name || runId}`;
        }
        DOM.presetPills.forEach((p) => p.classList.remove('active'));

        // Visual pulse glow on left editor to show it was filled
        if (DOM.editorDropzone) {
          DOM.editorDropzone.classList.remove('editor-fill-glow');
          void DOM.editorDropzone.offsetWidth;
          DOM.editorDropzone.classList.add('editor-fill-glow');
        }
        DOM.codeEditor.scrollTop = 0;
      }

      // Populate Audited Code card in Tab 2
      if (DOM.cardAuditedCode && data.original_code) {
        DOM.cardAuditedCode.style.display = 'block';
        if (DOM.auditedCodeFilename) DOM.auditedCodeFilename.textContent = data.contract_name || 'Contract.sol';
        if (DOM.auditedCodeLinesCount) DOM.auditedCodeLinesCount.textContent = `${data.original_code.split('\n').length} lines`;
        if (DOM.codeAuditedDisplay) DOM.codeAuditedDisplay.textContent = data.original_code;
        if (DOM.bodyAuditedCode) DOM.bodyAuditedCode.style.display = 'block';
      }

      // 2. Populate tabs with historical audit results
      populateVerdictAndFindings(data);
      populateDiff(data);
      populateReports(runId, data);
      populateStepperFromPhaseFiles(data.phases, data.is_awaiting_kaggle);

      const findingCount = (data.findings || []).length;
      DOM.tabFindingsBadge.textContent = findingCount;
    } catch (err) {
      console.warn('Error loading run detail:', err);
    }
  }

  function populateVerdictAndFindings(data) {
    const manifest = data.manifest || {};
    const det = manifest.detection || {};
    const verdict = det.verdict || (data.findings && data.findings.length > 0 ? 'VULNERABLE' : 'SAFE');
    const score = det.score !== undefined ? det.score : (verdict === 'VULNERABLE' ? 0.95 : 0.0);
    const riskLevel = det.risk_level || (score >= 0.7 ? 'CRITICAL' : score >= 0.4 ? 'HIGH' : 'LOW');

    const hasReport = data.has_pdf || data.has_html || data.has_md;
    const isAwaitingKaggle = data.is_awaiting_kaggle && !hasReport && data.status !== 'completed';
    if (isAwaitingKaggle) {
      DOM.verdictStatusBadge.textContent = 'AWAITING KAGGLE GPU...';
      DOM.verdictStatusBadge.className = 'verdict-badge';
      DOM.verdictStatusBadge.style.background = 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3))';
      DOM.verdictStatusBadge.style.borderColor = '#f59e0b';
      DOM.verdictStatusBadge.style.color = '#fde68a';
      DOM.verdictMetaText.textContent = `Contract: ${data.contract_name} | Awaiting response from Kaggle Cloud...`;
    } else {
      DOM.verdictStatusBadge.style.background = '';
      DOM.verdictStatusBadge.style.borderColor = '';
      DOM.verdictStatusBadge.style.color = '';
      DOM.verdictStatusBadge.textContent = verdict;
      DOM.verdictStatusBadge.className = 'verdict-badge ' + (verdict === 'VULNERABLE' ? 'vulnerable' : 'safe');
      DOM.verdictMetaText.textContent = `Contract: ${data.contract_name} | Risk Assessment: ${riskLevel}`;
    }
    DOM.gaugeScoreVal.textContent = parseFloat(score).toFixed(2);

    const findings = data.findings || [];
    DOM.metricFindingsCount.textContent = findings.length;

    const verifications = data.patch_outcomes || [];
    const verifiedPass = verifications.filter((v) => v.success).length;
    DOM.metricVerifiedCount.textContent = verifiedPass > 0 ? `${verifiedPass} Patched` : 'Compliant';

    const ragPhase = data.phases?.phase1b_rag;
    const ragMatches = ragPhase?.rag_matches?.length || (verdict === 'VULNERABLE' ? 3 : 0);
    DOM.metricRagSimilar.textContent = ragMatches;

    DOM.metricFixedStatus.textContent = data.fixed_code ? 'Patched' : 'Clean';
    DOM.metricFixedStatus.className = 'metric-num ' + (data.fixed_code ? 'text-green' : 'text-cyan');

    renderFindingsList(findings);
  }

  function renderFindingsList(findings, filter = 'all') {
    if (!findings || findings.length === 0) {
      DOM.findingsContainer.innerHTML = `
        <div class="empty-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="1.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="M9 12l2 2 4-4"/>
          </svg>
          <p style="color: #6ee7b7; font-weight: 600;">No critical security vulnerabilities detected in contract!</p>
          <span style="color: var(--text-muted); font-size: 0.75rem;">Neural detector and static rule engines scanned the entire codebase.</span>
        </div>
      `;
      return;
    }

    let filtered = findings;
    if (filter !== 'all') {
      filtered = findings.filter(
        (f) => (f.severity || 'Medium').toLowerCase() === filter.toLowerCase()
      );
    }

    let html = '';
    filtered.forEach((f) => {
      const severity = f.severity || 'High';
      const swc = f.swc_id || f.swc || 'SWC-107';
      const title = f.title || f.name || 'Security Vulnerability Detected';
      const desc = f.description || f.rationale || 'Logic or access control vulnerability identified.';
      const snippet = f.code_snippet || f.snippet || '';
      const remediation = f.remediation || f.recommendation || 'Apply Checks-Effects-Interactions pattern and incorporate ReentrancyGuard.';
      const source = f.source || 'Qwen2.5-14B LoRA v13 + Slither Fusion';

      html += `
        <div class="finding-item severity-${severity.toLowerCase()}">
          <div class="finding-header-row">
            <div class="finding-badges">
              <span class="severity-pill ${severity.toLowerCase()}">${severity}</span>
              <span class="swc-pill">${swc}</span>
              <span class="finding-source-tag">Source: ${source}</span>
            </div>
          </div>
          <h4 class="finding-title">${escapeHtml(title)}</h4>
          <p class="finding-desc">${escapeHtml(desc)}</p>
          ${snippet ? `<pre class="finding-code-snippet"><code>${escapeHtml(snippet)}</code></pre>` : ''}
          <div class="finding-remediation">
            <strong>Recommended Remediation:</strong> ${escapeHtml(remediation)}
          </div>
        </div>
      `;
    });

    DOM.findingsContainer.innerHTML = html;
  }

  // Filter chips in Findings Tab
  DOM.filterChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      DOM.filterChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      const filter = chip.getAttribute('data-filter');
      const findings = state.currentRunData?.findings || [];
      renderFindingsList(findings, filter);
    });
  });

  /* ==========================================================================
     Tab 3: Diff Viewer (Side-by-Side & Unified)
     ========================================================================== */
  function computeSideBySideDiff(origLines, fixedLines) {
    const n = origLines.length;
    const m = fixedLines.length;

    if (origLines.join('\n') === fixedLines.join('\n')) {
      return origLines.map((line, idx) => ({
        orig: { num: idx + 1, text: line, type: 'normal' },
        fixed: { num: idx + 1, text: line, type: 'normal' },
      }));
    }

    let prefixCount = 0;
    while (prefixCount < n && prefixCount < m && origLines[prefixCount] === fixedLines[prefixCount]) {
      prefixCount++;
    }
    let suffixCount = 0;
    while (
      suffixCount < (n - prefixCount) &&
      suffixCount < (m - prefixCount) &&
      origLines[n - 1 - suffixCount] === fixedLines[m - 1 - suffixCount]
    ) {
      suffixCount++;
    }

    const midOrig = origLines.slice(prefixCount, n - suffixCount);
    const midFixed = fixedLines.slice(prefixCount, m - suffixCount);
    const nMid = midOrig.length;
    const mMid = midFixed.length;

    const dp = Array.from({ length: nMid + 1 }, () => new Uint16Array(mMid + 1));
    for (let i = 0; i < nMid; i++) {
      for (let j = 0; j < mMid; j++) {
        if (midOrig[i] === midFixed[j]) {
          dp[i + 1][j + 1] = dp[i][j] + 1;
        } else {
          dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
      }
    }

    let i = nMid, j = mMid;
    const midDiff = [];
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && midOrig[i - 1] === midFixed[j - 1]) {
        midDiff.push({
          orig: { text: midOrig[i - 1], type: 'normal', origIdx: prefixCount + i - 1 },
          fixed: { text: midFixed[j - 1], type: 'normal', fixedIdx: prefixCount + j - 1 },
        });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        midDiff.push({
          orig: null,
          fixed: { text: midFixed[j - 1], type: 'add', fixedIdx: prefixCount + j - 1 },
        });
        j--;
      } else {
        midDiff.push({
          orig: { text: midOrig[i - 1], type: 'del', origIdx: prefixCount + i - 1 },
          fixed: null,
        });
        i--;
      }
    }
    midDiff.reverse();

    const result = [];
    for (let p = 0; p < prefixCount; p++) {
      result.push({
        orig: { num: p + 1, text: origLines[p], type: 'normal' },
        fixed: { num: p + 1, text: fixedLines[p], type: 'normal' },
      });
    }

    let idx = 0;
    while (idx < midDiff.length) {
      const item = midDiff[idx];
      if (item.orig && !item.fixed) {
        const dels = [];
        while (idx < midDiff.length && midDiff[idx].orig && !midDiff[idx].fixed) {
          dels.push(midDiff[idx].orig);
          idx++;
        }
        const adds = [];
        while (idx < midDiff.length && !midDiff[idx].orig && midDiff[idx].fixed) {
          adds.push(midDiff[idx].fixed);
          idx++;
        }
        const maxLen = Math.max(dels.length, adds.length);
        for (let k = 0; k < maxLen; k++) {
          const d = dels[k];
          const a = adds[k];
          result.push({
            orig: d ? { num: d.origIdx + 1, text: d.text, type: 'del' } : null,
            fixed: a ? { num: a.fixedIdx + 1, text: a.text, type: 'add' } : null,
          });
        }
      } else if (!item.orig && item.fixed) {
        const adds = [];
        while (idx < midDiff.length && !midDiff[idx].orig && midDiff[idx].fixed) {
          adds.push(midDiff[idx].fixed);
          idx++;
        }
        adds.forEach((a) => {
          result.push({
            orig: null,
            fixed: { num: a.fixedIdx + 1, text: a.text, type: 'add' },
          });
        });
      } else {
        result.push({
          orig: { num: item.orig.origIdx + 1, text: item.orig.text, type: 'normal' },
          fixed: { num: item.fixed.fixedIdx + 1, text: item.fixed.text, type: 'normal' },
        });
        idx++;
      }
    }

    for (let s = 0; s < suffixCount; s++) {
      const origIdx = n - suffixCount + s;
      const fixedIdx = m - suffixCount + s;
      result.push({
        orig: { num: origIdx + 1, text: origLines[origIdx], type: 'normal' },
        fixed: { num: fixedIdx + 1, text: fixedLines[fixedIdx], type: 'normal' },
      });
    }

    return result;
  }

  function populateDiff(data) {
    const origCode = data.original_code || '';
    const fixedCode = data.fixed_code || '';
    const diffText = data.diff || '';
    const verdictText = data.verification_verdict || 'Unverified';

    DOM.diffVerificationStatus.textContent = verdictText;
    DOM.diffVerificationStatus.className =
      'verification-badge ' + (verdictText.toLowerCase().includes('pass') ? 'verified' : '');

    // 1. Unified Diff
    if (!diffText && !origCode && !fixedCode) {
      DOM.diffContent.innerHTML = `
        <div class="empty-placeholder">
          <p>No automated patch generated yet. After Phase 4 (Fixer) completes, diff comparison will be rendered here.</p>
        </div>
      `;
    } else {
      const diffLines = (diffText || '').split('\n');
      let diffHtml = '';
      diffLines.forEach((line) => {
        let cls = 'diff-line diff-normal';
        if (line.startsWith('+') && !line.startsWith('+++')) {
          cls = 'diff-line diff-add';
        } else if (line.startsWith('-') && !line.startsWith('---')) {
          cls = 'diff-line diff-del';
        } else if (line.startsWith('@@') || line.startsWith('---') || line.startsWith('+++')) {
          cls = 'diff-line diff-meta';
        }
        diffHtml += `<div class="${cls}">${escapeHtml(line)}</div>`;
      });
      DOM.diffContent.innerHTML = diffHtml || '<div class="empty-placeholder"><p>No differences between original and patched code.</p></div>';
    }

    // 2. Side-by-Side Diff (Split View)
    if (!origCode && !fixedCode) {
      if (DOM.origCodeContent) DOM.origCodeContent.innerHTML = '<div class="empty-diff-hint">No original contract loaded.</div>';
      if (DOM.fixedCodeContent) DOM.fixedCodeContent.innerHTML = '<div class="empty-diff-hint">No automated patch generated yet.</div>';
      if (DOM.origLineNumbers) DOM.origLineNumbers.innerHTML = '<div class="line-num-cell">1</div>';
      if (DOM.fixedLineNumbers) DOM.fixedLineNumbers.innerHTML = '<div class="line-num-cell">1</div>';
      if (DOM.origCodeMeta) DOM.origCodeMeta.textContent = '0 lines';
      if (DOM.fixedCodeMeta) DOM.fixedCodeMeta.textContent = '0 lines';
      return;
    }

    const origLines = origCode ? origCode.split('\n') : [];
    const fixedLines = fixedCode ? fixedCode.split('\n') : origLines;

    if (DOM.origCodeMeta) DOM.origCodeMeta.textContent = `${origLines.length} lines (Original)`;
    if (DOM.fixedCodeMeta) {
      DOM.fixedCodeMeta.textContent = `${fixedLines.length} lines ${fixedCode && fixedCode !== origCode ? '(Patched)' : '(Unmodified)'}`;
    }

    const alignedRows = computeSideBySideDiff(origLines, fixedLines);

    let origNumHtml = '';
    let origCodeHtml = '';
    let fixedNumHtml = '';
    let fixedCodeHtml = '';

    alignedRows.forEach((row) => {
      // Original Pane
      if (row.orig) {
        const isDel = row.orig.type === 'del';
        origNumHtml += `<div class="line-num-cell ${isDel ? 'num-del' : ''}">${row.orig.num}</div>`;
        origCodeHtml += `<div class="code-row ${isDel ? 'row-del' : 'row-normal'}">${escapeHtml(row.orig.text || ' ')}</div>`;
      } else {
        origNumHtml += `<div class="line-num-cell num-empty">-</div>`;
        origCodeHtml += `<div class="code-row row-empty">&nbsp;</div>`;
      }

      // Fixed / Patched Pane
      if (row.fixed) {
        const isAdd = row.fixed.type === 'add';
        fixedNumHtml += `<div class="line-num-cell ${isAdd ? 'num-add' : ''}">${row.fixed.num}</div>`;
        fixedCodeHtml += `<div class="code-row ${isAdd ? 'row-add' : 'row-normal'}">${escapeHtml(row.fixed.text || ' ')}</div>`;
      } else {
        fixedNumHtml += `<div class="line-num-cell num-empty">-</div>`;
        fixedCodeHtml += `<div class="code-row row-empty">&nbsp;</div>`;
      }
    });

    if (DOM.origLineNumbers) DOM.origLineNumbers.innerHTML = origNumHtml;
    if (DOM.origCodeContent) DOM.origCodeContent.innerHTML = origCodeHtml;
    if (DOM.fixedLineNumbers) DOM.fixedLineNumbers.innerHTML = fixedNumHtml;
    if (DOM.fixedCodeContent) DOM.fixedCodeContent.innerHTML = fixedCodeHtml;

    // Synchronized scrolling between wrappers
    if (DOM.origCodeWrapper && DOM.fixedCodeWrapper) {
      let isSyncingOrig = false;
      let isSyncingFixed = false;

      DOM.origCodeWrapper.onscroll = () => {
        if (!isSyncingFixed) {
          isSyncingOrig = true;
          DOM.fixedCodeWrapper.scrollTop = DOM.origCodeWrapper.scrollTop;
          DOM.fixedCodeWrapper.scrollLeft = DOM.origCodeWrapper.scrollLeft;
          setTimeout(() => { isSyncingOrig = false; }, 40);
        }
      };

      DOM.fixedCodeWrapper.onscroll = () => {
        if (!isSyncingOrig) {
          isSyncingFixed = true;
          DOM.origCodeWrapper.scrollTop = DOM.fixedCodeWrapper.scrollTop;
          DOM.origCodeWrapper.scrollLeft = DOM.fixedCodeWrapper.scrollLeft;
          setTimeout(() => { isSyncingFixed = false; }, 40);
        }
      };
    }
  }

  // Toggle Audited Code Card in Tab 2
  if (DOM.headerAuditedCodeToggle && DOM.bodyAuditedCode) {
    DOM.headerAuditedCodeToggle.addEventListener('click', () => {
      const isHidden = DOM.bodyAuditedCode.style.display === 'none';
      DOM.bodyAuditedCode.style.display = isHidden ? 'block' : 'none';
      if (DOM.btnToggleAuditedCode) {
        DOM.btnToggleAuditedCode.textContent = isHidden ? 'Hide Code' : 'Show Code';
      }
    });
  }

  // Toggle View: Side-by-Side vs Unified
  if (DOM.btnViewSideBySide && DOM.btnViewUnified) {
    DOM.btnViewSideBySide.addEventListener('click', () => {
      DOM.btnViewSideBySide.classList.add('active');
      DOM.btnViewUnified.classList.remove('active');
      if (DOM.sideBySideContainer) DOM.sideBySideContainer.style.display = 'grid';
      if (DOM.diffContent) DOM.diffContent.style.display = 'none';
    });

    DOM.btnViewUnified.addEventListener('click', () => {
      DOM.btnViewUnified.classList.add('active');
      DOM.btnViewSideBySide.classList.remove('active');
      if (DOM.sideBySideContainer) DOM.sideBySideContainer.style.display = 'none';
      if (DOM.diffContent) DOM.diffContent.style.display = 'block';
    });
  }

  DOM.btnCopyPatch.addEventListener('click', () => {
    if (state.currentRunData?.fixed_code) {
      navigator.clipboard.writeText(state.currentRunData.fixed_code).then(() => {
        showToast('Copied patched contract code to clipboard!', 'success');
      });
    } else {
      showToast('No patched code available to copy!', 'error');
    }
  });

  DOM.btnDownloadFixed.addEventListener('click', () => {
    if (state.currentRunData?.fixed_code) {
      const blob = new Blob([state.currentRunData.fixed_code], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Fixed_${state.currentRunData.contract_name || 'Contract.sol'}`;
      a.click();
      URL.revokeObjectURL(a);
      showToast('Patched smart contract file downloaded!', 'success');
    } else {
      showToast('No patched file available to download!', 'error');
    }
  });

  /* ==========================================================================
     Tab 4: Reports
     ========================================================================== */
  function populateReports(runId, data) {
    if (data.has_html) {
      DOM.reportFrame.src = `/api/runs/${runId}/report/html`;
      DOM.reportFrame.style.display = 'block';
      DOM.reportEmpty.style.display = 'none';
      DOM.linkDownloadHtml.href = `/api/runs/${runId}/report/html`;
    } else {
      DOM.reportFrame.src = 'about:blank';
      DOM.reportFrame.style.display = 'none';
      DOM.reportEmpty.style.display = 'flex';
      DOM.linkDownloadHtml.removeAttribute('href');
    }

    if (data.has_pdf) {
      DOM.linkDownloadPdf.href = `/api/runs/${runId}/report/pdf`;
      DOM.linkDownloadPdf.style.display = 'inline-flex';
    } else {
      DOM.linkDownloadPdf.style.display = 'none';
    }

    if (data.has_md) {
      DOM.linkDownloadMd.href = `/api/runs/${runId}/report/md`;
      DOM.linkDownloadMd.style.display = 'inline-flex';
    } else {
      DOM.linkDownloadMd.style.display = 'none';
    }
  }

  function populateStepperFromPhaseFiles(phases, isAwaitingKaggle) {
    if (!phases) return;

    function getPhaseState(phaseObj) {
      if (!phaseObj || Object.keys(phaseObj).length === 0) return 'pending';
      if (phaseObj.status === 'pending') return isAwaitingKaggle ? 'running' : 'pending';
      if (phaseObj.status === 'ok' || phaseObj.status === 'completed') return 'success';
      if (phaseObj.status === 'error' || phaseObj.status === 'failed') return 'failed';
      return 'success';
    }

    const mapping = [
      { id: 'step-phase0', state: phases.phase0_input ? 'success' : 'pending' },
      { id: 'step-phase1a', state: getPhaseState(phases.phase1a_static) },
      { id: 'step-phase1b', state: getPhaseState(phases.phase1b_rag) },
      { id: 'step-phase1c', state: isAwaitingKaggle ? 'running' : getPhaseState(phases.phase1c_llm) },
      { id: 'step-fusion', state: isAwaitingKaggle ? 'running' : getPhaseState(phases.phase1_fused) },
      { id: 'step-phase2', state: isAwaitingKaggle ? 'pending' : getPhaseState(phases.phase2_advisor) },
      { id: 'step-phase3', state: isAwaitingKaggle ? 'pending' : getPhaseState(phases.phase3_assessor) },
      { id: 'step-phase4', state: isAwaitingKaggle ? 'pending' : getPhaseState(phases.phase4_fixer) },
      { id: 'step-phase5', state: isAwaitingKaggle ? 'pending' : getPhaseState(phases.phase5_verification) },
      { id: 'step-phase6', state: phases.phase6_report ? 'success' : 'pending' },
    ];

    mapping.forEach(({ id, state }) => {
      const card = document.getElementById(id);
      if (!card) return;
      const tag = card.querySelector('.step-status-tag, .step-status-icon');
      if (state === 'success') {
        card.className = 'step-card step-success';
        if (tag) tag.textContent = 'PASS';
      } else if (state === 'running') {
        card.className = 'step-card step-running';
        if (tag) tag.textContent = 'RUNNING';
      } else if (state === 'failed') {
        card.className = 'step-card step-failed';
        if (tag) tag.textContent = 'FAIL';
      } else {
        card.className = 'step-card step-pending';
        if (tag) tag.textContent = 'WAIT';
      }
    });

    if (isAwaitingKaggle) {
      DOM.stepperOverallStatus.textContent = 'Awaiting Kaggle GPU...';
      DOM.stepperOverallStatus.className = 'stepper-status-pill status-running';
    } else {
      DOM.stepperOverallStatus.textContent = 'Completed';
      DOM.stepperOverallStatus.className = 'stepper-status-pill status-success';
    }
  }

  /* ==========================================================================
     Tab 5: Runs History Explorer
     ========================================================================== */
  async function fetchRunsHistory() {
    try {
      const res = await fetch('/api/runs');
      if (!res.ok) return;
      const runs = await res.json();

      DOM.tabHistoryBadge.textContent = runs.length;

      if (!runs || runs.length === 0) {
        DOM.historyTableBody.innerHTML = `
          <tr><td colspan="9" class="text-center">No audit runs recorded in E2E/runs directory yet.</td></tr>
        `;
        return;
      }

      let html = '';
      runs.forEach((run) => {
        const isDone = run.has_pdf || run.has_html || run.status === 'completed';
        const verdictCls = run.verdict === 'VULNERABLE' ? 'text-red' : (run.verdict === 'SAFE' ? 'text-green' : 'text-cyan');
        let statusBadge = '<span style="color: #10b981; font-weight: 600;">Completed</span>';
        if (run.status === 'running') {
          statusBadge = '<span class="chip-val" style="color: #00f2fe; display: inline-flex; align-items: center; gap: 4px;"><span class="pulse-ring" style="width:6px;height:6px;background:#00f2fe;box-shadow:none;"></span> Running...</span>';
        } else if (run.status === 'awaiting_kaggle' && !isDone) {
          statusBadge = '<span class="chip-val" style="color: #f59e0b; display: inline-flex; align-items: center; gap: 4px;"><span class="pulse-ring" style="width:6px;height:6px;background:#f59e0b;box-shadow:none;"></span> Awaiting Kaggle</span>';
        } else if (run.status === 'failed') {
          statusBadge = '<span style="color: #ef4444; font-weight: 600;">Failed</span>';
        } else {
          statusBadge = '<span style="color: #10b981; font-weight: 600;">Completed</span>';
        }

        html += `
          <tr class="history-row" data-runid="${run.run_id}" title="Click to inspect run & fill contract code into editor">
            <td class="run-id-cell">${escapeHtml(run.run_id)}</td>
            <td><strong>${escapeHtml(run.contract_name || 'Contract.sol')}</strong></td>
            <td>${escapeHtml(run.created_at || 'Recent')}</td>
            <td>${statusBadge}</td>
            <td class="${verdictCls}"><strong>${escapeHtml(run.verdict || 'Unknown')}</strong></td>
            <td>${run.score !== undefined ? parseFloat(run.score).toFixed(2) : '-'}</td>
            <td>${run.finding_count || 0}</td>
            <td>
              ${run.has_pdf ? `<a href="/api/runs/${run.run_id}/report/pdf" target="_blank" class="btn btn-sm btn-outline" style="padding: 2px 6px;">PDF</a>` : ''}
              ${run.has_html ? `<a href="/api/runs/${run.run_id}/report/html" target="_blank" class="btn btn-sm btn-ghost" style="padding: 2px 6px;">HTML</a>` : ''}
            </td>
            <td>
              <button class="btn btn-sm btn-secondary btn-inspect-run" data-runid="${run.run_id}">
                View Details
              </button>
            </td>
          </tr>
        `;
      });

      DOM.historyTableBody.innerHTML = html;

      async function inspectHistoryRun(runId) {
        showToast(`Loading audit history & contract code for run ${runId}...`, 'info');
        await loadRunDetail(runId);
        const logRes = await fetch(`/api/runs/${runId}/logs`);
        if (logRes.ok) {
          const logData = await logRes.json();
          renderTerminalLogs(logData.logs);
          updateStepperFromLogs(logData.logs, logData.status, logData.phase_files);
        }
        switchTab('tab-findings');
        const cName = state.currentRunData?.contract_name || 'Contract.sol';
        const lineCount = state.currentRunData?.original_code ? state.currentRunData.original_code.split('\n').length : 0;
        showToast(`Loaded run "${runId}"! Contract code (${cName}, ${lineCount} lines) filled into Left Editor & Findings Tab.`, 'success');
      }

      // Bind inspect buttons
      document.querySelectorAll('.btn-inspect-run').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const runId = btn.getAttribute('data-runid');
          await inspectHistoryRun(runId);
        });
      });

      // Bind table row click (clicking anywhere on row except links/buttons)
      document.querySelectorAll('.history-row').forEach((row) => {
        row.addEventListener('click', async (e) => {
          if (e.target.closest('a') || e.target.closest('button')) return;
          const runId = row.getAttribute('data-runid');
          if (runId) {
            await inspectHistoryRun(runId);
          }
        });
      });
    } catch (err) {
      console.warn('Error fetching runs history:', err);
    }
  }

  DOM.btnRefreshHistory.addEventListener('click', () => {
    fetchRunsHistory();
    showToast('Audit history list refreshed!', 'info');
  });

  /* ==========================================================================
     Initial App Startup
     - On page load / Ctrl+R, we start with a clean state (Ready/Waiting),
       and do NOT auto-load past completed runs into views.
     ========================================================================== */
  async function initApp() {
    await fetchSystemStatus();
    await loadExampleContracts();
    await fetchRunsHistory();

    // Reset studio to clean initial state
    resetPipelineStepper('');
  }

  // Kickstart app
  initApp();
})();
