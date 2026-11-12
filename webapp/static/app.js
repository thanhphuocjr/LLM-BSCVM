(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // Five user-facing steps; each groups the pipeline phases whose log tags it covers.
  const STEPS = [
    { label: 'Phân tích', text: 'Đang phân tích mã', tags: ['[PHASE 0]'] },
    { label: 'Phát hiện', text: 'Đang tìm lỗ hổng', tags: ['[PHASE 1A]', '[PHASE 1B]', '[PHASE 1C]', '[PHASE 1 FUSED]'] },
    { label: 'Đánh giá', text: 'Đang đánh giá rủi ro', tags: ['[PHASE 2]', '[PHASE 3]'] },
    { label: 'Vá lỗi', text: 'Đang tạo bản vá', tags: ['[PHASE 4]', '[PHASE 5]'] },
    { label: 'Báo cáo', text: 'Đang viết báo cáo', tags: ['[PHASE 6]'] },
  ];

  const SEVERITY = {
    critical: 'Nghiêm trọng',
    high: 'Cao',
    medium: 'Trung bình',
    low: 'Thấp',
    informational: 'Thông tin',
    info: 'Thông tin',
    none: 'Không',
    unknown: 'Chưa rõ',
  };
  const SEVERITY_RANK = { critical: 0, high: 1, medium: 2, low: 3 };
  const SOURCES = { static: 'Static', rag: 'RAG', slither: 'Slither', llm_detector: 'Qwen' };

  const state = {
    system: null,
    examples: [],
    runs: [],
    runId: null,
    data: null,
    running: false,
    startedAt: 0,
    poll: null,
    clock: null,
    ticking: false,
    nameLocked: false,
  };

  const el = {
    tabs: document.querySelectorAll('.tab'),
    views: document.querySelectorAll('.view'),
    backendBtn: $('backendBtn'),
    backendPop: $('backendPop'),
    backendDot: $('backendDot'),
    backendNote: $('backendNote'),
    editorCard: $('editorCard'),
    fileName: $('fileName'),
    lineCount: $('lineCount'),
    samplesBtn: $('samplesBtn'),
    samplesPop: $('samplesPop'),
    fileInput: $('fileInput'),
    clearBtn: $('clearBtn'),
    gutter: $('gutter'),
    code: $('code'),
    runBtn: $('runBtn'),
    stopBtn: $('stopBtn'),
    progress: $('progress'),
    steps: $('steps'),
    progressText: $('progressText'),
    progressTime: $('progressTime'),
    result: $('result'),
    logCard: $('logCard'),
    logText: $('logText'),
    runList: $('runList'),
    toast: $('toast'),
  };

  /* ---------- Helpers ---------- */

  // Escape ký tự HTML (& < > " ') trước khi chèn dữ liệu vào innerHTML - chống XSS từ mã/kết quả người dùng.
  const esc = (value) =>
    String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

  // Ép giá trị về số; rỗng/không hợp lệ -> null (để ẩn ô điểm thay vì hiện NaN).
  const num = (value) => (value === null || value === undefined || value === '' || !Number.isFinite(+value) ? null : +value);

  // Kẹp số vào [0, 1] (vẽ thanh điểm rủi ro).
  const clamp01 = (value) => Math.max(0, Math.min(1, value));

  // Tạo URL API của một run: /api/runs/<run_id><path>.
  const runUrl = (runId, path = '') => `/api/runs/${encodeURIComponent(runId)}${path}`;

  let toastTimer;
  // Hiện thông báo nhỏ góc màn hình trong ~2.6 giây (isError = màu lỗi).
  function toast(message, isError = false) {
    el.toast.textContent = message;
    el.toast.className = 'toast show' + (isError ? ' error' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.remove('show'), 2600);
  }

  // Gọi API (fetch) và parse JSON; HTTP lỗi -> ném Error với thông báo "detail" của server.
  async function getJson(url, options) {
    const res = await fetch(url, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Lỗi ${res.status}`);
    return body;
  }

  // Lấy thời điểm của một run: created_at, không có thì đọc từ tiền tố run_id "YYYYmmdd-HHMMSS".
  function runDate(run) {
    const date = run.created_at ? new Date(run.created_at) : null;
    if (date && !Number.isNaN(date.getTime())) return date;
    const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(run.run_id || '');
    return m ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) : null;
  }

  // Định dạng ngày giờ kiểu Việt Nam (dd/mm/yyyy hh:mm).
  function formatDate(date) {
    if (!date) return '';
    return date.toLocaleString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  // Đổi mili giây thành "phút:giây" cho đồng hồ chạy.
  function formatElapsed(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  // Tải một đoạn text về máy thành file (dùng cho nút "Tải về" bản vá).
  function downloadText(text, fileName) {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------- Views & popovers ---------- */

  // Chuyển tab giao diện ("audit" / "history"); mở tab lịch sử thì tải lại danh sách run.
  function show(view) {
    el.tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.view === view)));
    el.views.forEach((v) => v.classList.toggle('active', v.id === view));
    if (view === 'history') loadHistory();
  }

  el.tabs.forEach((t) => t.addEventListener('click', () => show(t.dataset.view)));

  // Đóng mọi popover (trạng thái máy chủ, danh sách mẫu).
  function closePops() {
    document.querySelectorAll('.pop').forEach((p) => { p.hidden = true; });
    [el.backendBtn, el.samplesBtn].forEach((b) => b.setAttribute('aria-expanded', 'false'));
  }

  // Gắn nút với popover: bấm để mở/đóng, gọi onOpen khi mở (vd tải trạng thái server).
  function bindPop(button, pop, onOpen) {
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = pop.hidden;
      closePops();
      if (!willOpen) return;
      pop.hidden = false;
      button.setAttribute('aria-expanded', 'true');
      if (onOpen) onOpen();
    });
    pop.addEventListener('click', (e) => e.stopPropagation());
  }

  document.addEventListener('click', closePops);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePops();
  });

  /* ---------- Backend ---------- */

  // Vẽ chấm trạng thái GPU server (xanh = sẵn sàng, đỏ = chưa kết nối) theo state.system.
  function renderBackend() {
    const s = state.system;
    let online = null;
    let note = 'Đang kiểm tra…';
    if (s) {
      online = Boolean(s.server_gpu_online);
      note = online ? `${s.server_gpu_info?.gpu_device || 'Máy chủ GPU'} · sẵn sàng` : 'Máy chủ GPU chưa kết nối';
    }
    el.backendDot.className = 'dot' + (online === null ? '' : online ? ' on' : ' off');
    el.backendNote.textContent = note;
  }

  // Gọi GET /api/system/status rồi vẽ lại trạng thái máy chủ.
  async function loadSystem() {
    try {
      state.system = await getJson('/api/system/status');
    } catch {
      state.system = null;
    }
    renderBackend();
  }

  bindPop(el.backendBtn, el.backendPop, loadSystem);

  /* ---------- Editor ---------- */

  // Cập nhật số dòng và cột đánh số dòng (gutter) của ô soạn mã.
  function updateEditor() {
    const text = el.code.value;
    const lines = text ? text.split('\n').length : 0;
    el.lineCount.textContent = lines ? `${lines} dòng` : '';
    let gutter = '';
    for (let i = 1; i <= Math.max(lines, 1); i++) gutter += `${i}\n`;
    el.gutter.textContent = gutter + '\n\n';
    el.gutter.scrollTop = el.code.scrollTop;
  }

  // Lấy tên contract cuối cùng trong mã (để đặt tên file tự động).
  function contractName(source) {
    const names = [...source.matchAll(/\b(?:abstract\s+)?contract\s+([A-Za-z_$][\w$]*)/g)];
    return names.length ? names[names.length - 1][1] : null;
  }

  // Tự đặt tên file "<Contract>.sol" theo mã, trừ khi người dùng đã tự sửa tên.
  function autoName() {
    if (state.nameLocked) return;
    const name = contractName(el.code.value);
    el.fileName.value = `${name || 'Contract'}.sol`;
  }

  // Nạp mã vào ô soạn (từ mẫu, file, lịch sử), đặt tên file, xoá kết quả cũ.
  function setCode(text, fileName) {
    el.code.value = text;
    el.code.scrollTop = 0;
    if (fileName) {
      el.fileName.value = fileName;
      state.nameLocked = true;
    } else {
      state.nameLocked = false;
      autoName();
    }
    updateEditor();
    clearResult();
  }

  // Code copied out of JSON datasets arrives as one escaped line ("...\n...").
  // Đổi mã dán từ dataset JSON (một dòng có \n, \t, \") về mã nhiều dòng bình thường.
  function unescapeCode(raw) {
    let text = raw.trim();
    if (/^(["`])[\s\S]*\1$/.test(text)) text = text.slice(1, -1);
    return text
      .replace(/\\r\\n|\\n/g, '\n')
      .replace(/\\t/g, '    ')
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'")
      .replace(/\\\\/g, '\\');
  }

  // Chèn text tại vị trí con trỏ (giữ được Undo của trình duyệt).
  function insertText(text) {
    el.code.focus();
    if (!document.execCommand('insertText', false, text)) {
      el.code.setRangeText(text, el.code.selectionStart, el.code.selectionEnd, 'end');
      el.code.dispatchEvent(new Event('input'));
    }
  }

  el.code.addEventListener('input', () => {
    updateEditor();
    autoName();
    if (state.data && !state.running) clearResult();
  });

  el.code.addEventListener('scroll', () => {
    el.gutter.scrollTop = el.code.scrollTop;
  });

  el.code.addEventListener('paste', (e) => {
    if (el.code.readOnly) return;
    const replacesAll = el.code.selectionStart === 0 && el.code.selectionEnd === el.code.value.length;
    if (replacesAll) state.nameLocked = false;
    const raw = e.clipboardData?.getData('text') || '';
    if (!raw.includes('\n') && raw.includes('\\n')) {
      e.preventDefault();
      insertText(unescapeCode(raw));
    }
  });

  el.code.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!state.running) startRun();
    } else if (e.key === 'Tab' && !el.code.readOnly) {
      e.preventDefault();
      insertText('    ');
    }
  });

  el.fileName.addEventListener('input', () => {
    state.nameLocked = true;
  });

  el.clearBtn.addEventListener('click', () => {
    setCode('');
    el.code.focus();
  });

  // Đọc file .sol người dùng chọn/kéo thả và nạp vào ô soạn.
  function readFile(file) {
    if (state.running) return;
    const reader = new FileReader();
    reader.onload = () => setCode(String(reader.result), file.name);
    reader.onerror = () => toast('Không đọc được file', true);
    reader.readAsText(file);
  }

  el.fileInput.addEventListener('change', () => {
    if (el.fileInput.files[0]) readFile(el.fileInput.files[0]);
    el.fileInput.value = '';
  });

  ['dragenter', 'dragover'].forEach((type) => {
    el.editorCard.addEventListener(type, (e) => {
      e.preventDefault();
      if (!state.running) el.editorCard.classList.add('drop');
    });
  });

  el.editorCard.addEventListener('dragleave', (e) => {
    if (!el.editorCard.contains(e.relatedTarget)) el.editorCard.classList.remove('drop');
  });

  el.editorCard.addEventListener('drop', (e) => {
    e.preventDefault();
    el.editorCard.classList.remove('drop');
    const file = e.dataTransfer.files[0];
    if (file) readFile(file);
  });

  /* ---------- Samples ---------- */

  // Dòng mô tả ngắn cho một contract mẫu (loại lỗi / "Không lỗi").
  function sampleNote(example) {
    if (/safe/i.test(example.vuln_type || '')) return 'Không lỗi';
    if (!example.vuln_type || example.vuln_type === 'Unknown') return `${example.line_count} dòng`;
    return example.vuln_type;
  }

  // Gọi GET /api/contracts/examples và vẽ danh sách contract mẫu trong popover "Mẫu".
  async function loadExamples() {
    try {
      state.examples = await getJson('/api/contracts/examples');
    } catch {
      state.examples = [];
    }
    el.samplesPop.innerHTML = state.examples.length
      ? state.examples
        .map((ex, i) => `<button class="pick" data-i="${i}"><b>${esc(ex.title)}</b><span>${esc(sampleNote(ex))}</span></button>`)
        .join('')
      : '<div class="empty">Không có mẫu</div>';
  }

  bindPop(el.samplesBtn, el.samplesPop);

  el.samplesPop.addEventListener('click', (e) => {
    const b = e.target.closest('.pick');
    if (!b) return;
    const ex = state.examples[+b.dataset.i];
    closePops();
    if (ex) setCode(ex.code, ex.file_name);
  });

  /* ---------- Running an audit ---------- */

  el.steps.innerHTML = STEPS.map(
    (s) => `<div class="step"><div class="step-bar"></div><div class="step-label">${s.label}</div></div>`
  ).join('');
  const stepEls = [...el.steps.children];

  // Tô thanh tiến độ 5 bước (Phân tích, Phát hiện, Đánh giá, Vá lỗi, Báo cáo).
  function setStep(current) {
    stepEls.forEach((node, i) => {
      node.classList.toggle('done', i < current);
      node.classList.toggle('active', i === current);
    });
  }

  // Suy ra bước hiện tại từ log: tìm tag "[PHASE x]" mới nhất xuất hiện (bảng STEPS ánh xạ phase -> bước).
  function currentStep(logs) {
    let index = 0;
    STEPS.forEach((s, i) => {
      if (s.tags.some((tag) => logs.includes(tag))) index = i;
    });
    return index;
  }

  // Bật/tắt chế độ "đang chạy": đổi nút Chạy/Dừng, khoá ô soạn và các nút.
  function setRunning(on) {
    state.running = on;
    el.runBtn.hidden = on;
    el.stopBtn.hidden = !on;
    el.stopBtn.disabled = false;
    el.code.readOnly = on;
    el.samplesBtn.disabled = on;
    el.clearBtn.disabled = on;
    el.fileInput.disabled = on;
  }

  // Hiện khung tiến độ với dòng trạng thái, đặt về bước 0.
  function showProgress(text) {
    el.progress.hidden = false;
    el.progress.classList.remove('failed');
    el.progressText.textContent = text;
    el.progressTime.textContent = '';
    setStep(0);
  }

  // Dừng các bộ đếm (poll log mỗi giây và đồng hồ).
  function stopTracking() {
    clearInterval(state.poll);
    clearInterval(state.clock);
    state.poll = null;
    state.clock = null;
  }

  // Bắt đầu theo dõi một run: đồng hồ mỗi giây + poll /logs mỗi giây (tick).
  function track(runId, startedAt) {
    stopTracking();
    state.runId = runId;
    state.startedAt = startedAt;
    // Cập nhật đồng hồ thời gian chạy.
    const updateClock = () => {
      el.progressTime.textContent = formatElapsed(Date.now() - state.startedAt);
    };
    updateClock();
    state.clock = setInterval(updateClock, 1000);
    state.poll = setInterval(tick, 1000);
    tick();
  }

  // Một lần poll GET /api/runs/<id>/logs: cập nhật bước tiến độ + log; nếu run kết thúc thì gọi finish().
  async function tick() {
    if (state.ticking || !state.running) return;
    state.ticking = true;
    const runId = state.runId;
    try {
      const d = await getJson(runUrl(runId, '/logs'));
      if (runId !== state.runId || !state.running) return;
      const logs = d.logs || '';
      const finished = d.status === 'completed' || d.status === 'failed' || d.status === 'cancelled';
      if (!finished) {
        const step = currentStep(logs);
        setStep(step);
        el.progressText.textContent = `${STEPS[step].text}…`;
      }
      if (!el.logCard.hidden) renderLog(logs);
      if (finished) await finish(d.status);
    } catch {
      // Transient network error: the next tick retries.
    } finally {
      state.ticking = false;
    }
  }

  // Kết thúc theo dõi: thành công -> mở kết quả (openRun); dừng/lỗi -> báo trạng thái và mở log.
  async function finish(status) {
    stopTracking();
    setRunning(false);
    if (status === 'completed') {
      el.progress.hidden = true;
      await openRun(state.runId);
      toast('Đã kiểm tra xong');
      return;
    }
    el.progress.classList.add('failed');
    if (status === 'cancelled') {
      el.progressText.textContent = 'Đã dừng';
    } else {
      el.progressText.textContent = 'Có lỗi khi chạy';
      openLog();
      toast('Có lỗi khi chạy', true);
    }
  }

  // Nút "Chạy": gửi POST /api/audit/submit với mã + tên file rồi track(run_id).
  async function startRun() {
    const code = el.code.value;
    if (!code.trim()) {
      toast('Hãy dán mã trước', true);
      el.code.focus();
      return;
    }
    clearResult();
    setRunning(true);
    showProgress('Đang gửi…');
    try {
      const body = await getJson('/api/audit/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          file_name: el.fileName.value.trim() || 'Contract.sol',
        }),
      });
      track(body.run_id, Date.now());
    } catch (err) {
      setRunning(false);
      el.progress.hidden = true;
      toast(err.message || 'Không gửi được', true);
    }
  }

  // Tiếp tục theo dõi một run đang chạy (vd tải lại trang khi audit chưa xong).
  async function resume(run) {
    try {
      const data = await getJson(runUrl(run.run_id));
      if (data.original_code) setCode(data.original_code, data.contract_name);
    } catch {
      // The run directory may not be readable yet; keep tracking anyway.
    }
    setRunning(true);
    showProgress('Đang chạy…');
    track(run.run_id, Date.now() - (run.elapsed_seconds || 0) * 1000);
  }

  el.runBtn.addEventListener('click', startRun);

  el.stopBtn.addEventListener('click', async () => {
    if (!state.running) return;
    el.stopBtn.disabled = true;
    try {
      await getJson(runUrl(state.runId, '/cancel'), { method: 'POST' });
      await finish('cancelled');
    } catch {
      // Already finished on the server: the next tick picks up the final state.
      el.stopBtn.disabled = false;
    }
  });

  /* ---------- Log ---------- */

  // Hiển thị log (tự cuộn xuống cuối nếu người dùng đang ở cuối).
  function renderLog(text) {
    const pre = el.logText;
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.textContent = text || 'Chưa có nhật ký';
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }

  // Mở khung nhật ký và tải log của run hiện tại.
  async function openLog() {
    el.logCard.hidden = false;
    if (!state.runId) {
      renderLog('');
      return;
    }
    try {
      const d = await getJson(runUrl(state.runId, '/logs'));
      renderLog(d.logs);
      el.logText.scrollTop = el.logText.scrollHeight;
    } catch {
      renderLog('Không tải được nhật ký');
    }
  }

  // Bật/tắt khung nhật ký.
  function toggleLog() {
    if (el.logCard.hidden) {
      openLog().then(() => el.logCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
    } else {
      el.logCard.hidden = true;
    }
  }

  /* ---------- Result ---------- */

  // Xoá kết quả đang hiển thị.
  function clearResult() {
    state.data = null;
    el.result.innerHTML = '';
    el.logCard.hidden = true;
    if (!state.running) el.progress.hidden = true;
  }

  // Đổi verdict (Vulnerable/Safe/Inconclusive) thành nhãn tiếng Việt + màu.
  function verdictOf(verdict) {
    switch (String(verdict || '').toLowerCase()) {
      case 'vulnerable': return { tone: 'bad', word: 'Có lỗ hổng' };
      case 'safe': return { tone: 'ok', word: 'An toàn' };
      case 'inconclusive': return { tone: 'warn', word: 'Chưa rõ' };
      default: return { tone: '', word: 'Chưa có kết quả' };
    }
  }

  // Đổi overall_verdict của Phase 5 (final_verification.decide_overall_verdict) thành nhãn bản vá:
  // "Đã vá" / "Chưa vá" (kèm lý do) / "Không cần" / "Chưa rõ" (kèm lý do).
  function patchStatus(data) {
    const raw = String(data.manifest?.verification?.overall_verdict || data.verification_verdict || '');
    const v = raw.toLowerCase();
    if (!v || v === 'pending') return { label: '—', rejected: false };
    if (v.includes('no patch required')) return { label: 'Không cần', rejected: false };
    if (v.includes('accepted')) return { label: 'Đã vá', rejected: false };
    if (v.includes('rejected')) {
      let sub = '';
      if (v.includes('compile')) sub = 'Không biên dịch được';
      else if (v.includes('behavioral')) sub = 'Sai hành vi';
      else if (v.includes('residual') || v.includes('introduced') || v.includes('still detect')) sub = 'Vẫn còn lỗ hổng';
      else if (v.includes('no effective')) sub = 'Chưa tạo được bản vá';
      return { label: 'Chưa vá', sub, rejected: true };
    }
    if (v.startsWith('inconclusive')) {
      let sub = '';
      if (v.includes('no patch was generated')) sub = 'Chưa tạo được bản vá';
      else if (v.includes('repair target')) sub = 'Không có hàm để vá';
      else if (v.includes('incomplete')) sub = 'Chưa chạy xong';
      else if (v.includes('in isolation')) sub = 'Không biên dịch độc lập được';
      else if (v.includes('slither')) sub = 'Slither chưa kết luận';
      return { label: 'Chưa rõ', sub, rejected: false };
    }
    return { label: raw, rejected: false };
  }

  // Đổi mức độ (critical/high/...) thành nhãn tiếng Việt.
  function severityLabel(value) {
    const key = String(value || 'unknown').toLowerCase();
    return { key, label: SEVERITY[key] || value };
  }

  // Vẽ danh sách lỗi: sắp theo mức độ rồi độ tin cậy; mỗi lỗi có hàm, dòng, mô tả, cách sửa, SWC, nguồn
  // (Static/RAG/Slither/Qwen), độ tin cậy, CVSS (từ Phase 3).
  function renderFindings(findings, assessments) {
    const sorted = [...findings].sort((a, b) => {
      const ra = SEVERITY_RANK[String(a.severity).toLowerCase()] ?? 9;
      const rb = SEVERITY_RANK[String(b.severity).toLowerCase()] ?? 9;
      return ra - rb || (num(b.confidence) || 0) - (num(a.confidence) || 0);
    });

    const rows = sorted
      .map((f) => {
        const sev = severityLabel(f.severity);
        const fn = String(f.unit_id || '').split('.').pop();
        const where = [fn && `${fn}()`, f.line && `dòng ${f.line}`].filter(Boolean).join(' · ');
        const assessment = assessments[f.finding_id] || {};
        const confidence = num(f.confidence);
        const cvss = num(assessment.cvss_score);
        const meta = [
          f.swc,
          (f.sources || []).length && `Nguồn: ${f.sources.map((s) => SOURCES[s] || s).join(', ')}`,
          confidence !== null && `Tin cậy ${confidence.toFixed(2)}`,
          cvss !== null && `CVSS ${cvss.toFixed(1)}`,
        ].filter(Boolean);
        const description = f.description || f.root_cause || '';
        const remediation = f.remediation || '';
        return `
          <div class="finding">
            <button class="finding-row" data-act="toggle">
              <span class="sev ${esc(sev.key)}">${esc(sev.label)}</span>
              <span class="finding-title">${esc(f.title || f.swc || 'Lỗ hổng')}</span>
              <span class="finding-where">${esc(where)}</span>
            </button>
            <div class="finding-detail">
              ${description ? `<p>${esc(description)}</p>` : ''}
              ${remediation ? `<div class="fix"><b>Cách sửa:</b> ${esc(remediation)}</div>` : ''}
              ${meta.length ? `<div class="finding-meta">${esc(meta.join(' · '))}</div>` : ''}
            </div>
          </div>`;
      })
      .join('');

    return `
      <div class="card section enter">
        <div class="section-head"><div class="section-title">Lỗi <span class="muted">${findings.length}</span></div></div>
        ${rows || '<div class="empty">Chưa xác định được vị trí lỗi</div>'}
      </div>`;
  }

  // Vẽ bản vá dạng diff (+ xanh / - đỏ) kèm nút Sao chép / Tải về.
  function renderDiff(diff, rejected) {
    const lines = diff.split('\n');
    const start = lines.findIndex((l) => l.startsWith('@@'));
    const body = (start >= 0 ? lines.slice(start) : lines)
      .map((line) => {
        const cls = line.startsWith('@@') ? 'hunk' : line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : '';
        return `<div class="${cls}">${esc(line) || ' '}</div>`;
      })
      .join('');
    return `
      <div class="card section enter">
        <div class="section-head">
          <div class="section-title">Bản vá${rejected ? ' <span class="muted">chưa đạt</span>' : ''}</div>
          <button class="btn ghost sm" data-act="copy">Sao chép</button>
          <button class="btn ghost sm" data-act="download">Tải về</button>
        </div>
        <pre class="diff">${body}</pre>
      </div>`;
  }

  // Vẽ toàn bộ kết quả một run: verdict, điểm fusion + vạch ngưỡng, số lỗi, mức rủi ro, trạng thái bản vá,
  // link báo cáo PDF/HTML, danh sách lỗi và diff bản vá.
  function renderResult(data) {
    const det = data.manifest?.detection || {};
    const fused = data.phases?.phase1_fused || {};
    const findings = data.findings || [];
    const verdict = verdictOf(det.verdict || fused.verdict);
    const score = num(det.score ?? fused.score);
    const threshold = num(fused.threshold);
    const risk = severityLabel(det.risk_level || fused.risk_level || 'none');
    const patch = patchStatus(data);

    const assessments = {};
    (data.phases?.phase3_assessor?.assessments || []).forEach((a) => {
      assessments[a.finding_id] = a;
    });

    const meter = score === null ? '' : `
      <div class="meter">
        <div class="meter-fill" style="width:${(clamp01(score) * 100).toFixed(1)}%"></div>
        ${threshold === null ? '' : `<div class="meter-mark" style="left:${(clamp01(threshold) * 100).toFixed(1)}%"><span>${threshold.toFixed(2)}</span></div>`}
      </div>`;

    const actions = [
      data.has_pdf && `<a class="btn sm" href="${runUrl(data.run_id, '/report/pdf')}">Tải PDF</a>`,
      data.has_html && `<a class="btn sm" href="${runUrl(data.run_id, '/report/html')}" target="_blank" rel="noopener">Xem báo cáo</a>`,
      '<button class="btn ghost sm" data-act="log">Nhật ký</button>',
    ].filter(Boolean).join('');

    let html = `
      <div class="result ${verdict.tone} enter">
        <div class="result-top">
          <div class="verdict">${verdict.word}</div>
          ${score === null ? '' : `<div class="score" title="Điểm rủi ro">${score.toFixed(2)}</div>`}
        </div>
        ${meter}
        <div class="facts">
          <div><div class="fact-label">Lỗi</div><div class="fact-value">${findings.length}</div></div>
          <div><div class="fact-label">Mức rủi ro</div><div class="fact-value">${esc(risk.label)}</div></div>
          <div>
            <div class="fact-label">Bản vá</div>
            <div class="fact-value">${esc(patch.label)}</div>
            ${patch.sub ? `<div class="fact-sub">${esc(patch.sub)}</div>` : ''}
          </div>
        </div>
        <div class="result-actions">${actions}</div>
      </div>`;

    if (findings.length || verdict.tone === 'bad') html += renderFindings(findings, assessments);
    if (data.diff) html += renderDiff(data.diff, patch.rejected);

    el.result.innerHTML = html;
  }

  // Tải chi tiết run (GET /api/runs/<id>) và hiển thị; fillEditor = nạp lại mã gốc vào ô soạn (mở từ lịch sử).
  async function openRun(runId, fillEditor = false) {
    try {
      const data = await getJson(runUrl(runId));
      if (fillEditor) {
        el.logCard.hidden = true;
        if (data.original_code) setCode(data.original_code, data.contract_name);
      }
      state.runId = runId;
      state.data = data;
      renderResult(data);
      el.result.firstElementChild?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch {
      toast('Không mở được kết quả', true);
    }
  }

  el.result.addEventListener('click', (e) => {
    const target = e.target.closest('[data-act]');
    if (!target) return;
    const act = target.dataset.act;
    const data = state.data;
    if (act === 'toggle') {
      target.parentElement.classList.toggle('open');
    } else if (act === 'log') {
      toggleLog();
    } else if (act === 'copy' && data?.fixed_code) {
      navigator.clipboard.writeText(data.fixed_code).then(
        () => toast('Đã sao chép'),
        () => toast('Không sao chép được', true)
      );
    } else if (act === 'download' && data?.fixed_code) {
      downloadText(data.fixed_code, `Fixed_${data.contract_name || 'Contract.sol'}`);
    }
  });

  el.progress.addEventListener('click', (e) => {
    if (e.target.closest('[data-act="log"]')) toggleLog();
  });

  /* ---------- History ---------- */

  // Nhãn trạng thái của một run trong danh sách lịch sử.
  function runVerdict(run) {
    if (run.status === 'running') return { tone: 'live', word: 'Đang chạy' };
    if (run.status === 'failed') return { tone: '', word: 'Lỗi chạy' };
    const v = verdictOf(run.verdict);
    return v.tone ? v : { tone: '', word: '—' };
  }

  // Tải và vẽ danh sách lịch sử audit (run đang chạy lên đầu, sau đó mới nhất trước).
  async function loadHistory() {
    if (!el.runList.innerHTML) el.runList.innerHTML = '<div class="empty">Đang tải…</div>';
    try {
      // Folders without any run artifacts are not audits.
      state.runs = (await getJson('/api/runs'))
        .filter((r) => r.status === 'running' || r.created_at || r.verdict !== 'Unknown')
        .sort((a, b) =>
          (b.status === 'running') - (a.status === 'running') ||
          (runDate(b)?.getTime() || 0) - (runDate(a)?.getTime() || 0));
    } catch {
      el.runList.innerHTML = '<div class="empty">Không tải được lịch sử</div>';
      return;
    }
    if (!state.runs.length) {
      el.runList.innerHTML = '<div class="empty">Chưa có lần kiểm tra nào</div>';
      return;
    }
    el.runList.innerHTML = state.runs
      .map((run) => {
        const v = runVerdict(run);
        const score = num(run.score);
        const showScore = v.tone && v.tone !== 'live' && score !== null;
        const meta = [formatDate(runDate(run)), run.status !== 'running' && `${run.finding_count || 0} lỗi`]
          .filter(Boolean)
          .join(' · ');
        return `
          <button class="run" data-id="${esc(run.run_id)}">
            <div class="run-main">
              <div class="run-name">${esc(run.contract_name || run.run_id)}</div>
              <div class="run-meta">${esc(meta)}</div>
            </div>
            <span class="run-verdict ${v.tone}">${v.word}</span>
            <span class="run-score">${showScore ? score.toFixed(2) : ''}</span>
          </button>`;
      })
      .join('');
  }

  el.runList.addEventListener('click', (e) => {
    const b = e.target.closest('.run');
    if (!b) return;
    const id = b.dataset.id;
    if (state.running) {
      if (id === state.runId) show('audit');
      else toast('Đang có lượt kiểm tra chạy', true);
      return;
    }
    const run = state.runs.find((r) => r.run_id === id);
    show('audit');
    if (run && run.status === 'running') resume(run);
    else openRun(id, true);
  });

  /* ---------- Start ---------- */

  // Khởi động trang: tab audit, trạng thái máy chủ, mẫu, và tự tiếp tục theo dõi run đang chạy (nếu có).
  async function init() {
    show('audit');
    renderBackend();
    updateEditor();
    loadSystem();
    loadExamples();
    try {
      state.runs = await getJson('/api/runs');
      const live = state.runs.find((r) => r.status === 'running');
      if (live && !state.running) resume(live);
    } catch {
      // History is optional at startup.
    }
  }

  init();
})();
